//! JoyCode Anthropic Provider Adapter
//!
//! 处理 JoyCode 的 Anthropic 适配器认证流程：
//! 1. 先调用 /api/saas/model-runtime/v1/models/prepare 获取 X-Model-Token
//! 2. 用 token 调用实际的 /api/saas/anthropic/v1/messages 接口

use super::{AuthInfo, ProviderAdapter};
use crate::provider::Provider;
use crate::proxy::error::ProxyError;
use crate::proxy::http_client;
use crate::proxy::json_canonical::short_sha256_hex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

pub(crate) const JOYCODE_VSCODE_CLIENT: &str = "VS Code";
pub(crate) const JOYCODE_VSCODE_CLIENT_VERSION: &str = "3.8.58";
pub(crate) const JOYCODE_DEFAULT_LANGUAGE: &str = "UNKNOWN";

/// JoyCode Anthropic 适配器
pub struct JoyCodeAnthropicAdapter {
    client: Client,
    token_cache: Arc<Mutex<HashMap<String, CachedToken>>>,
}

#[derive(Clone)]
struct CachedToken {
    token: String,
    chat_id: String,
    expires_at: SystemTime,
}

#[derive(Clone, Debug)]
pub(crate) struct JoyCodePreparedToken {
    pub(crate) token: String,
    pub(crate) chat_id: String,
}

#[derive(Serialize)]
struct PrepareRequest {
    model: String,
    #[serde(rename = "chatId")]
    chat_id: String,
    stream: bool,
    client: String,
    #[serde(rename = "clientVersion")]
    client_version: String,
    language: String,
    #[serde(rename = "orgFullName")]
    org_full_name: String,
}

#[derive(Deserialize)]
struct PrepareResponse {
    code: i32,
    msg: Option<String>,
    data: Option<PrepareData>,
}

#[derive(Deserialize)]
struct PrepareData {
    token: Option<String>,
    #[serde(rename = "chatId")]
    chat_id: Option<String>,
    #[serde(rename = "tokenStatus")]
    token_status: Option<String>,
    #[serde(rename = "nextPollAt")]
    next_poll_at: Option<String>,
    #[serde(rename = "expireAt")]
    expire_at: Option<String>,
}

impl JoyCodeAnthropicAdapter {
    pub fn new() -> Self {
        Self {
            client: http_client::get_direct(),
            token_cache: shared_token_cache(),
        }
    }

    /// 获取或刷新 X-Model-Token
    pub async fn get_model_prepare(
        &self,
        base_url: &str,
        model: &str,
        stream: bool,
        pt_key: &str,
        login_type: &str,
        tenant: &str,
        org_full_name: &str,
    ) -> Result<JoyCodePreparedToken, ProxyError> {
        let cache_key = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            base_url.trim_end_matches('/'),
            model,
            stream,
            login_type,
            tenant,
            org_full_name,
            short_sha256_hex(pt_key.as_bytes())
        );

        // 检查缓存
        {
            let cache = self.token_cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                if cached.expires_at > SystemTime::now() {
                    log::debug!("[JoyCode] Using cached X-Model-Token for {}", model);
                    return Ok(JoyCodePreparedToken {
                        token: cached.token.clone(),
                        chat_id: cached.chat_id.clone(),
                    });
                }
            }
        }

        // 调用 prepare 接口。部分 JoyCode 模型会先返回 WAITING，需要按同一 chatId 轮询。
        let prepare_url = format!("{}/api/saas/model-runtime/v1/models/prepare", base_url);
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis();
        let request_chat_id = format!("proxy-{}", timestamp);
        let mut last_status = String::new();
        let (token, chat_id, expires_at) = {
            let mut token = None;
            let mut chat_id = request_chat_id.clone();
            let mut expires_at = None;
            for attempt in 1..=60 {
                let request_body = PrepareRequest {
                    model: model.to_string(),
                    chat_id: chat_id.clone(),
                    stream,
                    client: JOYCODE_VSCODE_CLIENT.to_string(),
                    client_version: JOYCODE_VSCODE_CLIENT_VERSION.to_string(),
                    language: JOYCODE_DEFAULT_LANGUAGE.to_string(),
                    org_full_name: org_full_name.to_string(),
                };

                log::debug!(
                    "[JoyCode] Calling prepare API: {} (attempt={attempt})",
                    prepare_url
                );

                let response = self
                    .client
                    .post(&prepare_url)
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .header("ptKey", pt_key)
                    .header("loginType", login_type)
                    .header("tenant", tenant)
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(|e| {
                        ProxyError::ForwardFailed(format!("prepare request failed: {}", e))
                    })?;

                let status = response.status();
                let response_text = response.text().await.unwrap_or_else(|_| String::from("{}"));

                log::debug!(
                    "[JoyCode] Prepare response: status={}, body={}",
                    status,
                    crate::proxy::sensitive_redaction::redact_sensitive_text(
                        &response_text[..response_text.len().min(200)]
                    )
                );

                let parsed: PrepareResponse =
                    serde_json::from_str(&response_text).map_err(|e| {
                        ProxyError::ForwardFailed(format!(
                            "prepare response parse failed: {} (body: {})",
                            e,
                            crate::proxy::sensitive_redaction::redact_sensitive_text(
                                &response_text[..response_text.len().min(200)]
                            )
                        ))
                    })?;

                if parsed.code != 0 || parsed.data.is_none() {
                    return Err(ProxyError::ConfigError(format!(
                        "prepare failed: code={}, msg={}",
                        parsed.code,
                        parsed.msg.unwrap_or_default()
                    )));
                }

                let data = parsed.data.ok_or_else(|| {
                    ProxyError::ConfigError("prepare returned no data".to_string())
                })?;
                if let Some(next_chat_id) = data.chat_id {
                    chat_id = next_chat_id;
                }
                if let Some(expire_at) = data.expire_at.as_deref().and_then(parse_expire_at) {
                    expires_at = Some(expire_at);
                }
                last_status = data.token_status.unwrap_or_else(|| "READY".to_string());
                token = data.token;
                if last_status.eq_ignore_ascii_case("READY") {
                    break;
                }

                let sleep_ms = data
                    .next_poll_at
                    .as_deref()
                    .and_then(next_poll_delay_ms)
                    .unwrap_or(1000)
                    .clamp(250, 3000);
                log::info!(
                    "[JoyCode] model token not ready: model={}, status={}, sleep_ms={}",
                    model,
                    last_status,
                    sleep_ms
                );
                tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
            }
            let token = token.ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "prepare returned no token after polling, status={last_status}"
                ))
            })?;
            if !last_status.eq_ignore_ascii_case("READY") {
                return Err(ProxyError::Timeout(format!(
                    "JoyCode model token not ready after polling: model={model}, status={last_status}"
                )));
            }
            (
                token,
                chat_id,
                expires_at.unwrap_or_else(|| SystemTime::now() + Duration::from_secs(45)),
            )
        };

        // 缓存 token。JoyCode 返回的模型 token 通常只有约 60 秒有效期，留 5 秒余量。
        {
            let mut cache = self.token_cache.lock().unwrap();
            cache.insert(
                cache_key,
                CachedToken {
                    token: token.clone(),
                    chat_id: chat_id.clone(),
                    expires_at,
                },
            );
        }

        log::info!("[JoyCode] X-Model-Token prepared for model: {}", model);
        Ok(JoyCodePreparedToken { token, chat_id })
    }
}

fn shared_token_cache() -> Arc<Mutex<HashMap<String, CachedToken>>> {
    static CACHE: OnceLock<Arc<Mutex<HashMap<String, CachedToken>>>> = OnceLock::new();
    CACHE
        .get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
        .clone()
}

fn next_poll_delay_ms(next_poll_at: &str) -> Option<u64> {
    let target = chrono::DateTime::parse_from_rfc3339(next_poll_at).ok()?;
    let now = chrono::Utc::now();
    let delta = target.with_timezone(&chrono::Utc) - now;
    Some(delta.num_milliseconds().max(0) as u64)
}

fn parse_expire_at(expire_at: &str) -> Option<SystemTime> {
    let target = chrono::DateTime::parse_from_rfc3339(expire_at)
        .ok()?
        .with_timezone(&chrono::Utc);
    let now = chrono::Utc::now();
    let ttl_ms = (target - now).num_milliseconds().saturating_sub(5_000);
    if ttl_ms <= 0 {
        return None;
    }
    Some(SystemTime::now() + Duration::from_millis(ttl_ms as u64))
}

impl ProviderAdapter for JoyCodeAnthropicAdapter {
    fn name(&self) -> &'static str {
        "joycode-anthropic"
    }

    fn extract_base_url(&self, provider: &Provider) -> Result<String, ProxyError> {
        provider
            .settings_config
            .get("baseURL")
            .or_else(|| provider.settings_config.get("base_url"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim_end_matches('/').to_string())
            .ok_or_else(|| ProxyError::ConfigError("JoyCode provider missing baseURL".to_string()))
    }

    fn extract_auth(&self, _provider: &Provider) -> Option<AuthInfo> {
        // JoyCode Anthropic 使用 prepare 流程，不需要提前提取 auth
        None
    }

    fn build_url(&self, base_url: &str, endpoint: &str) -> String {
        format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        )
    }

    fn get_auth_headers(
        &self,
        _auth: &AuthInfo,
    ) -> Result<Vec<(http::HeaderName, http::HeaderValue)>, ProxyError> {
        // 认证头在 prepare 后动态生成，这里返回空
        Ok(vec![])
    }
}

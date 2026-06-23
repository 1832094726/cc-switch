//! 供应商连通性检查服务（reachability）
//!
//! 仅探测供应商 `base_url` 是否可达，**不发送真实大模型请求**：
//! - 收到任意 HTTP 响应（200/4xx/5xx）即判定"可达"（端口通、网关存活）；
//! - 仅 DNS / 连接被拒 / TLS / 超时等网络级错误判定"不可达"；
//! - 延迟 = 收到响应头的耗时（TTFB，真实往返）。
//!
//! ## 设计取舍：可达 ≠ 配置正确
//!
//! 本检查刻意不验证鉴权或模型，因此不会被第三方供应商的鉴权拦截 / 模型校验
//! 误判为"不可用"。代价是它无法告诉你鉴权对不对、模型存不存在。
//!
//! ## 与故障转移的关系（重要不变量）
//!
//! 连通性检查 **绝不** 触碰故障转移熔断器：一个返回 403/401 的供应商在本检查里
//! 算"可达"，但它对真实流量是坏的。熔断器只由 `proxy/forwarder.rs` 转发真实流量
//! 的成败驱动（被动）。两者职责分离——可达性回答"能不能到"，真实流量回答"能不能用"。

use reqwest::header::HeaderValue;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::app_config::AppType;
use crate::error::AppError;
use crate::provider::Provider;
use crate::proxy::providers::{get_adapter, ClaudeAdapter, ProviderAdapter};
use bytes::Bytes;
use futures::StreamExt;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use crate::proxy::gemini_url::{normalize_gemini_model_id, resolve_gemini_native_url};
use crate::proxy::providers::copilot_auth;
use crate::proxy::providers::transform::anthropic_to_openai;
use crate::proxy::providers::transform_gemini::anthropic_to_gemini;
use crate::proxy::providers::transform_responses::anthropic_to_responses;
use crate::proxy::providers::{AuthInfo, AuthStrategy};

/// 健康状态枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Operational,
    Degraded,
    Failed,
}

/// 连通性检查配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamCheckConfig {
    /// 单次探测超时（秒）
    pub timeout_secs: u64,
    /// 超时类失败的最大重试次数
    pub max_retries: u32,
    /// 降级阈值（毫秒）：可达但 TTFB 超过该值判定为"较慢"
    pub degraded_threshold_ms: u64,
    /// 测试用 prompt（仅 devin 流式检查使用）
    pub test_prompt: String,
    /// Claude 默认测试模型（仅 devin 流式检查使用）
    pub claude_model: String,
    /// Codex 默认测试模型（仅 devin 流式检查使用）
    pub codex_model: String,
    /// Gemini 默认测试模型（仅 devin 流式检查使用）
    pub gemini_model: String,
}

impl Default for StreamCheckConfig {
    fn default() -> Self {
        // 可达性探测打的是 base_url 的小请求（仅读响应头），不等待模型生成，故超时远小于
        // 旧的真实请求检查（45s → 8s）；降级阈值沿用旧尺度 6000ms——探测 TTFB 一般远低于
        // 此，仅在确实很慢时才标"较慢"，避免把 1 秒多的正常延迟误判为降级。
        Self {
            timeout_secs: 8,
            test_prompt: "Hi".to_string(),
            claude_model: "claude-sonnet-4-20250514".to_string(),
            codex_model: "gpt-5".to_string(),
            gemini_model: "gemini-2.5-flash".to_string(),
            max_retries: 1,
            degraded_threshold_ms: 6000,
        }
    }
}

/// 连通性检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamCheckResult {
    pub status: HealthStatus,
    pub success: bool,
    pub message: String,
    pub response_time_ms: Option<u64>,
    pub http_status: Option<u16>,
    /// 保留字段以兼容 `stream_check_logs` 表结构；连通性检查恒为空串。
    pub model_used: String,
    pub tested_at: i64,
    pub retry_count: u32,
    /// 细粒度错误分类；连通性检查不再细分，恒为 None。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_category: Option<String>,
}

/// 连通性检查服务
pub struct StreamCheckService;


#[derive(Debug, Clone)]
struct DevinCatalogRoute {
    request_model: String,
    upstream_model: String,
    endpoint: String,
    base_url: String,
    api_key: String,
    auth_header: Option<String>,
    headers: Option<serde_json::Map<String, Value>>,
    responses_mode: Option<String>,
    responses_fast_mode: bool,
}

impl DevinCatalogRoute {
    fn auth_strategy(&self) -> AuthStrategy {
        match self.auth_header.as_deref() {
            Some("x-api-key") => AuthStrategy::Anthropic,
            _ => AuthStrategy::Bearer,
        }
    }
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn bool_field(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|value| {
            value.as_bool().or_else(|| {
                value.as_str().map(str::trim).map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "true" | "1" | "yes" | "on"
                    )
                })
            })
        })
    })
}

fn header_fields(value: &Value, keys: &[&str]) -> serde_json::Map<String, Value> {
    let Some(headers) = keys
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_object))
    else {
        return serde_json::Map::new();
    };

    headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            let value = value
                .as_str()
                .map(str::to_string)
                .or_else(|| value.as_bool().map(|value| value.to_string()))
                .or_else(|| value.as_i64().map(|value| value.to_string()))
                .or_else(|| value.as_u64().map(|value| value.to_string()))
                .or_else(|| value.as_f64().map(|value| value.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                return None;
            }
            Some((name.to_string(), Value::String(value.to_string())))
        })
        .collect()
}

fn merge_header_fields(
    base: &mut serde_json::Map<String, Value>,
    patch: serde_json::Map<String, Value>,
) {
    for (name, value) in patch {
        if let Some(existing) = base
            .keys()
            .find(|existing| existing.eq_ignore_ascii_case(&name))
            .cloned()
        {
            base.insert(existing, value);
        } else {
            base.insert(name, value);
        }
    }
}


impl StreamCheckService {
    /// 执行连通性检查（仅对超时类失败重试）。
    ///
    /// `base_url_override`：用于 Copilot 等需要从 OAuth 管理器动态解析端点的供应商，
    /// 由命令层预先解析后传入；其余供应商传 `None`，由本服务从 `settings_config` 提取。
    pub async fn check_with_retry(
        app_type: &AppType,
        provider: &Provider,
        config: &StreamCheckConfig,
        base_url_override: Option<String>,
    ) -> Result<StreamCheckResult, AppError> {
        let effective = Self::merge_provider_config(provider, config);

        let mut last_result: Option<StreamCheckResult> = None;
        for attempt in 0..=effective.max_retries {
            let start = Instant::now();
            let result = Self::check_once(
                app_type,
                provider,
                &effective,
                base_url_override.clone(),
                start,
            )
            .await?;

            if result.success {
                return Ok(StreamCheckResult {
                    retry_count: attempt,
                    ..result
                });
            }

            // 仅超时 / abort 类网络抖动值得重试；连接被拒、DNS 失败等立即返回。
            if Self::should_retry(&result.message) && attempt < effective.max_retries {
                last_result = Some(result);
                continue;
            }
            return Ok(StreamCheckResult {
                retry_count: attempt,
                ..result
            });
        }

        Ok(last_result.unwrap_or_else(|| StreamCheckResult {
            status: HealthStatus::Failed,
            success: false,
            message: "Check failed".to_string(),
            response_time_ms: None,
            http_status: None,
            model_used: String::new(),
            tested_at: chrono::Utc::now().timestamp(),
            retry_count: effective.max_retries,
            error_category: None,
        }))
    }

    /// 合并供应商单独配置（`meta.testConfig`，仅当 `enabled`）与全局配置。
    fn merge_provider_config(provider: &Provider, global: &StreamCheckConfig) -> StreamCheckConfig {
        let tc = provider
            .meta
            .as_ref()
            .and_then(|m| m.test_config.as_ref())
            .filter(|tc| tc.enabled);

        match tc {
            Some(tc) => StreamCheckConfig {
                timeout_secs: tc.timeout_secs.unwrap_or(global.timeout_secs),
                max_retries: tc.max_retries.unwrap_or(global.max_retries),
                degraded_threshold_ms: tc
                    .degraded_threshold_ms
                    .unwrap_or(global.degraded_threshold_ms),
                test_prompt: tc
                    .test_prompt
                    .clone()
                    .unwrap_or_else(|| global.test_prompt.clone()),
                claude_model: tc
                    .test_model
                    .clone()
                    .unwrap_or_else(|| global.claude_model.clone()),
                codex_model: tc
                    .test_model
                    .clone()
                    .unwrap_or_else(|| global.codex_model.clone()),
                gemini_model: tc
                    .test_model
                    .clone()
                    .unwrap_or_else(|| global.gemini_model.clone()),
            },
            None => global.clone(),
        }
    }

    /// 单次连通性探测。
    async fn check_once(
        app_type: &AppType,
        provider: &Provider,
        config: &StreamCheckConfig,
        base_url_override: Option<String>,
        start: Instant,
    ) -> Result<StreamCheckResult, AppError> {
        let base_url = match base_url_override {
            Some(b) => b,
            None => Self::resolve_base_url(app_type, provider)?,
        };

        let client = crate::proxy::http_client::get();
        let timeout = std::time::Duration::from_secs(config.timeout_secs);
        let ua = Self::custom_user_agent(provider);

        let result = Self::probe_reachability(&client, &base_url, timeout, ua).await;
        let response_time = start.elapsed().as_millis() as u64;
        Ok(Self::build_result(
            result,
            response_time,
            config.degraded_threshold_ms,
        ))
    }

    /// 解析供应商 `base_url`。
    ///
    /// 连通性探测只需打到 base（origin 或用户配置的 base 路径）即可——任何 HTTP
    /// 响应都证明端口可达，因此无需像旧的真实请求检查那样解析具体 API 路径
    /// （`/v1/messages` vs `/chat/completions` vs `:streamGenerateContent`）。
    ///
    /// 官方供应商（`category == "official"`）base_url 故意留空（走客户端默认/OAuth 端点），
    /// 没有 cc-switch 能可靠探测的目标——这类供应商的连通检测按钮在前端已隐藏
    /// （见 `ProviderCard.tsx`），故此处对其提取失败直接报错即可，不做官方端点回退。
    fn resolve_base_url(app_type: &AppType, provider: &Provider) -> Result<String, AppError> {
        match app_type {
            // 累加模式应用的 settings_config 结构与 Claude/Codex/Gemini 不同，
            // 不走 adapter，直接按各自约定提取 base_url。
            AppType::OpenCode => {
                let npm = Self::extract_opencode_npm(provider);
                Self::resolve_opencode_base_url(provider, npm.as_deref())
            }
            AppType::OpenClaw => Self::extract_openclaw_base_url(provider),
            AppType::Hermes => Self::extract_hermes_base_url(provider),
            AppType::ClaudeDesktop => ClaudeAdapter::new()
                .extract_base_url(provider)
                .map_err(|e| AppError::Message(format!("Failed to extract base_url: {e}"))),
            _ => get_adapter(app_type)
                .extract_base_url(provider)
                .map_err(|e| AppError::Message(format!("Failed to extract base_url: {e}"))),
        }
    }

    /// 轻量可达性探测：GET `base_url`，收到任意 HTTP 响应即可达。
    ///
    /// - `send()` 在收到响应头时即返回，故计时天然是 TTFB；不读 body。
    /// - reqwest 对任何 HTTP 状态码都返回 `Ok`，只有网络级错误进 `Err`——
    ///   这正是"任何响应都算可达、只有连不上才算失败"的语义。
    async fn probe_reachability(
        client: &Client,
        base_url: &str,
        timeout: std::time::Duration,
        custom_ua: Option<HeaderValue>,
    ) -> Result<u16, AppError> {
        let url = base_url.trim();
        if url.is_empty() {
            return Err(AppError::Message("base_url 为空".to_string()));
        }

        let mut req = client
            .get(url)
            .timeout(timeout)
            .header("accept", "*/*")
            .header("accept-encoding", "identity");
        // 复用供应商自定义 UA（部分网关按 UA 白名单放行），与转发路径口径一致。
        if let Some(ua) = custom_ua {
            req = req.header("user-agent", ua);
        }

        match req.send().await {
            Ok(resp) => Ok(resp.status().as_u16()),
            Err(e) => Err(Self::map_request_error(e)),
        }
    }

    /// 将探测原始结果包装成 `StreamCheckResult`。
    fn build_result(
        result: Result<u16, AppError>,
        response_time: u64,
        degraded_threshold_ms: u64,
    ) -> StreamCheckResult {
        let tested_at = chrono::Utc::now().timestamp();
        match result {
            Ok(status) => StreamCheckResult {
                status: Self::determine_status(response_time, degraded_threshold_ms),
                success: true,
                message: "Reachable".to_string(),
                response_time_ms: Some(response_time),
                http_status: Some(status),
                model_used: String::new(),
                tested_at,
                retry_count: 0,
                error_category: None,
            },
            Err(e) => StreamCheckResult {
                status: HealthStatus::Failed,
                success: false,
                message: e.to_string(),
                response_time_ms: Some(response_time),
                http_status: None,
                model_used: String::new(),
                tested_at,
                retry_count: 0,
                error_category: None,
            },
        }
    }

    fn determine_status(latency_ms: u64, threshold: u64) -> HealthStatus {
        if latency_ms <= threshold {
            HealthStatus::Operational
        } else {
            HealthStatus::Degraded
        }
    }

    fn should_retry(msg: &str) -> bool {
        let lower = msg.to_lowercase();
        lower.contains("timeout") || lower.contains("abort") || lower.contains("timed out")
    }

    fn map_request_error(e: reqwest::Error) -> AppError {
        if e.is_timeout() {
            AppError::Message("Request timeout".to_string())
        } else if e.is_connect() {
            AppError::Message(format!("Connection failed: {e}"))
        } else {
            AppError::Message(e.to_string())
        }
    }

    /// Provider 级自定义 User-Agent（`meta.customUserAgent`），与转发路径共用单一口径：
    /// trim、空串视为未设置、非法值静默忽略（返回 `None`）。
    fn custom_user_agent(provider: &Provider) -> Option<HeaderValue> {
        provider
            .meta
            .as_ref()
            .and_then(|meta| meta.custom_user_agent_header().ok().flatten())
    }

    // ===== 各应用 base_url 提取（settings_config 结构互不相同）=====

    /// OpenClaw: `{ baseUrl, apiKey, api, ... }`（camelCase）
    fn extract_openclaw_base_url(provider: &Provider) -> Result<String, AppError> {
        provider
            .settings_config
            .get("baseUrl")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                AppError::localized(
                    "openclaw_base_url_missing",
                    "OpenClaw 供应商缺少 baseUrl",
                    "OpenClaw provider is missing `baseUrl`",
                )
            })
    }

    /// Hermes: `{ base_url, api_key, api_mode }`（snake_case）
    fn extract_hermes_base_url(provider: &Provider) -> Result<String, AppError> {
        provider
            .settings_config
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                AppError::localized(
                    "hermes_base_url_missing",
                    "Hermes 供应商缺少 base_url",
                    "Hermes provider is missing `base_url`",
                )
            })
    }

    /// OpenCode: `{ npm, options: { baseURL, apiKey }, ... }`
    ///
    /// 用户未显式填 `options.baseURL` 时，按 `npm`（AI SDK 包）回退到包自带默认端点。
    /// `@ai-sdk/openai-compatible` 无默认端点，必须显式填。
    fn resolve_opencode_base_url(
        provider: &Provider,
        npm: Option<&str>,
    ) -> Result<String, AppError> {
        if let Some(explicit) = Self::extract_opencode_base_url(provider) {
            return Ok(explicit);
        }

        let fallback = match npm {
            Some("@ai-sdk/openai") => Some("https://api.openai.com/v1"),
            Some("@ai-sdk/anthropic") => Some("https://api.anthropic.com"),
            Some("@ai-sdk/google") => Some("https://generativelanguage.googleapis.com"),
            _ => None,
        };

        fallback.map(|s| s.to_string()).ok_or_else(|| {
            AppError::localized(
                "opencode_base_url_missing",
                "OpenCode 供应商缺少 options.baseURL，且当前 SDK 包没有默认端点",
                "OpenCode provider is missing `options.baseURL` and the SDK package has no default endpoint",
            )
        })
    }

    fn extract_opencode_base_url(provider: &Provider) -> Option<String> {
        provider
            .settings_config
            .get("options")
            .and_then(|v| v.get("baseURL"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn extract_opencode_npm(provider: &Provider) -> Option<String> {
        provider
            .settings_config
            .get("npm")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    // ─── Devin / JoyCode stream check (ported from pre-merge) ───

    fn build_joycode_color_gateway_url(function_id: &str) -> String {
        Self::build_joycode_color_gateway_url_with_time(
            function_id,
            chrono::Utc::now().timestamp_millis(),
        )
    }

    fn build_joycode_color_gateway_url_with_time(function_id: &str, timestamp_ms: i64) -> String {
        const JOYCODE_COLOR_GATEWAY_ORIGIN: &str = "https://api-ai.jd.com";
        let params = vec![
            ("appid", "joycode_ide".to_string()),
            ("functionId", function_id.to_string()),
            ("t", timestamp_ms.to_string()),
        ];
        let sign = Self::joycode_color_gateway_sign(&params);
        let mut url = url::Url::parse(JOYCODE_COLOR_GATEWAY_ORIGIN)
            .expect("JoyCode color gateway origin is valid");
        url.set_path("/api");
        {
            let mut query = url.query_pairs_mut();
            for (key, value) in &params {
                query.append_pair(key, value);
            }
            query.append_pair("sign", &sign);
        }
        url.to_string()
    }

    fn build_stream_check_result(
        result: Result<(u16, String), AppError>,
        response_time: u64,
        degraded_threshold_ms: u64,
        model_tested: &str,
    ) -> StreamCheckResult {
        let tested_at = chrono::Utc::now().timestamp();
        match result {
            Ok((status_code, model)) => StreamCheckResult {
                status: Self::determine_status(response_time, degraded_threshold_ms),
                success: true,
                message: "Check succeeded".to_string(),
                response_time_ms: Some(response_time),
                http_status: Some(status_code),
                model_used: model,
                tested_at,
                retry_count: 0,
                error_category: None,
            },
            Err(e) => {
                let (http_status, message, error_category) = match &e {
                    AppError::HttpStatus { status, body } => {
                        let category = Self::detect_error_category(*status, body);
                        (
                            Some(*status),
                            Self::classify_http_status(*status).to_string(),
                            category.map(|s| s.to_string()),
                        )
                    }
                    _ => (None, e.to_string(), None),
                };
                StreamCheckResult {
                    status: HealthStatus::Failed,
                    success: false,
                    message,
                    response_time_ms: Some(response_time),
                    http_status,
                    model_used: model_tested.to_string(),
                    tested_at,
                    retry_count: 0,
                    error_category,
                }
            }
        }
    }

    async fn check_claude_stream(
        client: &Client,
        base_url: &str,
        auth: &AuthInfo,
        model: &str,
        test_prompt: &str,
        timeout: std::time::Duration,
        provider: &Provider,
        claude_api_format_override: Option<&str>,
        extra_headers: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<(u16, String), AppError> {
        let base = base_url.trim_end_matches('/');
        let is_github_copilot = auth.strategy == AuthStrategy::GitHubCopilot;

        // Detect api_format: meta.api_format > settings_config.api_format > default "anthropic"
        let api_format = provider
            .meta
            .as_ref()
            .and_then(|m| m.api_format.as_deref())
            .or_else(|| {
                provider
                    .settings_config
                    .get("api_format")
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("anthropic");

        let effective_api_format = claude_api_format_override.unwrap_or(api_format);

        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);
        let is_openai_chat = effective_api_format == "openai_chat";
        let is_openai_responses = effective_api_format == "openai_responses";
        let is_gemini_native = effective_api_format == "gemini_native";
        let url = Self::resolve_claude_stream_url(
            base,
            auth.strategy,
            effective_api_format,
            is_full_url,
            model,
        );

        let max_tokens = if is_openai_responses { 16 } else { 1 };

        // Build from Anthropic-native shape first, then convert for configured targets.
        let anthropic_body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{ "role": "user", "content": test_prompt }],
            "stream": true
        });
        // Codex OAuth (ChatGPT Plus/Pro 反代) 需要 store:false + include 标记，
        // 否则 Stream Check 会和生产路径一样被服务端 400 拒绝。
        let is_codex_oauth = provider.is_codex_oauth();
        let codex_fast_mode = provider.codex_fast_mode_enabled();

        let body = if is_openai_responses {
            anthropic_to_responses(
                anthropic_body,
                Some(&provider.id),
                is_codex_oauth,
                codex_fast_mode,
            )
            .map_err(|e| AppError::Message(format!("Failed to build test request: {e}")))?
        } else if is_gemini_native {
            anthropic_to_gemini(anthropic_body)
                .map_err(|e| AppError::Message(format!("Failed to build test request: {e}")))?
        } else if is_openai_chat {
            anthropic_to_openai(anthropic_body)
                .map_err(|e| AppError::Message(format!("Failed to build test request: {e}")))?
        } else {
            anthropic_body
        };

        let mut request_builder = client.post(&url);

        if is_github_copilot {
            // 生成请求追踪 ID
            let request_id = uuid::Uuid::new_v4().to_string();
            request_builder = request_builder
                .header("authorization", format!("Bearer {}", auth.api_key))
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("accept-encoding", "identity")
                .header("user-agent", copilot_auth::COPILOT_USER_AGENT)
                .header("editor-version", copilot_auth::COPILOT_EDITOR_VERSION)
                .header(
                    "editor-plugin-version",
                    copilot_auth::COPILOT_PLUGIN_VERSION,
                )
                .header(
                    "copilot-integration-id",
                    copilot_auth::COPILOT_INTEGRATION_ID,
                )
                .header("x-github-api-version", copilot_auth::COPILOT_API_VERSION)
                // 260401 新增copilot 的关键 headers
                .header("openai-intent", "conversation-agent")
                .header("x-initiator", "user")
                .header("x-interaction-type", "conversation-agent")
                .header("x-vscode-user-agent-library-version", "electron-fetch")
                .header("x-request-id", &request_id)
                .header("x-agent-task-id", &request_id);
        } else if is_gemini_native {
            request_builder = match auth.strategy {
                AuthStrategy::GoogleOAuth => {
                    let token = auth.access_token.as_ref().unwrap_or(&auth.api_key);
                    request_builder
                        .header("authorization", format!("Bearer {token}"))
                        .header("x-goog-api-client", "GeminiCLI/1.0")
                        .header("content-type", "application/json")
                        .header("accept", "text/event-stream")
                        .header("accept-encoding", "identity")
                }
                _ => request_builder
                    .header("x-goog-api-key", &auth.api_key)
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .header("accept-encoding", "identity"),
            };
        } else if is_openai_chat || is_openai_responses {
            // OpenAI-compatible targets: Bearer auth + SSE headers only
            request_builder = request_builder
                .header("authorization", format!("Bearer {}", auth.api_key))
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("accept-encoding", "identity");
        } else {
            // Anthropic native: full Claude CLI headers
            let os_name = Self::get_os_name();
            let arch_name = Self::get_arch_name();

            // 鉴权头复用 ClaudeAdapter::get_auth_headers，与代理路径（forwarder）保持单一真理来源。
            // - AuthStrategy::Anthropic  → x-api-key
            // - AuthStrategy::ClaudeAuth → Authorization: Bearer
            // - AuthStrategy::Bearer     → Authorization: Bearer
            // 避免之前"无条件 Bearer + 条件 x-api-key 双发"导致的假阴性 / auth conflict。
            let auth_headers = ClaudeAdapter::new()
                .get_auth_headers(auth)
                .map_err(|e| AppError::Message(format!("stream check 构造鉴权头失败: {e}")))?;
            for (name, value) in auth_headers {
                request_builder = request_builder.header(name, value);
            }

            request_builder = request_builder
                // Anthropic required headers
                .header("anthropic-version", "2023-06-01")
                .header(
                    "anthropic-beta",
                    "claude-code-20250219,interleaved-thinking-2025-05-14",
                )
                .header("anthropic-dangerous-direct-browser-access", "true")
                // Content type headers
                .header("content-type", "application/json")
                .header("accept", "application/json")
                .header("accept-encoding", "identity")
                .header("accept-language", "*")
                // Client identification headers
                .header("user-agent", "claude-cli/2.1.2 (external, cli)")
                .header("x-app", "cli")
                // x-stainless SDK headers (dynamic local system info)
                .header("x-stainless-lang", "js")
                .header("x-stainless-package-version", "0.70.0")
                .header("x-stainless-os", os_name)
                .header("x-stainless-arch", arch_name)
                .header("x-stainless-runtime", "node")
                .header("x-stainless-runtime-version", "v22.20.0")
                .header("x-stainless-retry-count", "0")
                .header("x-stainless-timeout", "600")
                // Other headers
                .header("sec-fetch-mode", "cors");
        }

        // 供应商自定义 headers 最后追加，允许覆盖内置默认值（例如 user-agent）
        if let Some(headers) = extra_headers {
            for (key, value) in headers {
                if let Some(v) = value.as_str() {
                    request_builder = request_builder.header(key.as_str(), v);
                }
            }
        }

        // Provider 级自定义 User-Agent（meta.customUserAgent）覆盖默认 UA，与 forwarder
        // 转发路径口径一致；Copilot 指纹 UA 不可被覆盖。
        if !is_github_copilot {
            if let Some(ua) = Self::custom_user_agent(provider) {
                request_builder = request_builder.header("user-agent", ua);
            }
        }

        let response = request_builder
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(Self::map_request_error)?;

        let status = response.status().as_u16();

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::http_status_error(status, error_text));
        }

        // 流式读取：只需首个 chunk
        let mut stream = response.bytes_stream();
        if let Some(chunk) = stream.next().await {
            match chunk {
                Ok(_) => Ok((status, model.to_string())),
                Err(e) => Err(AppError::Message(format!("Stream read failed: {e}"))),
            }
        } else {
            Err(AppError::Message("No response data received".to_string()))
        }
    }

    pub async fn check_devin_catalog(
        provider: &Provider,
        config: &StreamCheckConfig,
    ) -> Vec<StreamCheckResult> {
        let effective_config = Self::merge_provider_config(provider, config);
        let routes = Self::devin_catalog_routes(provider);
        if routes.is_empty() {
            return vec![StreamCheckResult {
                status: HealthStatus::Failed,
                success: false,
                message: "No Devin model routes configured".to_string(),
                response_time_ms: None,
                http_status: None,
                model_used: String::new(),
                tested_at: chrono::Utc::now().timestamp(),
                retry_count: 0,
                error_category: None,
            }];
        }

        let client = crate::proxy::http_client::get();
        let timeout = std::time::Duration::from_secs(effective_config.timeout_secs);
        let mut results = Vec::with_capacity(routes.len());
        for route in routes {
            let start = Instant::now();
            log::info!(
                "[StreamCheck][Devin] testing provider={} request_model={} upstream_model={} endpoint={} base_url={}",
                provider.id,
                route.request_model,
                route.upstream_model,
                route.endpoint,
                route.base_url
            );
            let result = Self::check_devin_route_once(
                &client,
                provider,
                &route,
                &effective_config.test_prompt,
                timeout,
            )
            .await;
            let response_time = start.elapsed().as_millis() as u64;
            let mut result = Self::build_stream_check_result(
                result,
                response_time,
                effective_config.degraded_threshold_ms,
                &format!("{} -> {}", route.request_model, route.upstream_model),
            );
            result.model_used = format!("{} -> {}", route.request_model, route.upstream_model);
            log::info!(
                "[StreamCheck][Devin] result provider={} model={} success={} status={:?} http_status={:?} message={}",
                provider.id,
                result.model_used,
                result.success,
                result.status,
                result.http_status,
                result.message
            );
            results.push(result);
        }

        results
    }

    async fn check_devin_chat_stream(
        client: &Client,
        provider: &Provider,
        route: &DevinCatalogRoute,
        test_prompt: &str,
        timeout: std::time::Duration,
    ) -> Result<(u16, String), AppError> {
        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);
        let urls = Self::resolve_codex_chat_stream_urls(&route.base_url, is_full_url);
        let (actual_model, reasoning_effort) = Self::parse_model_with_effort(&route.upstream_model);

        let os_name = Self::get_os_name();
        let arch_name = Self::get_arch_name();
        let user_agent = Self::custom_user_agent(provider).unwrap_or_else(|| {
            reqwest::header::HeaderValue::from_str(&format!(
                "codex_cli_rs/0.80.0 ({os_name} 15.7.2; {arch_name}) Terminal"
            ))
            .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("codex_cli_rs/0.80.0"))
        });

        let mut body = json!({
            "model": actual_model,
            "messages": [{ "role": "user", "content": test_prompt }],
            "max_tokens": 10,
            "stream": true
        });

        if let Some(effort) = reasoning_effort {
            if crate::proxy::providers::transform::supports_reasoning_effort(&actual_model) {
                body["reasoning_effort"] = json!(effort);
            }
        }

        let mut last_error: Option<AppError> = None;
        for (index, url) in urls.iter().enumerate() {
            let mut request_builder = client
                .post(url)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("accept-encoding", "identity")
                .header("user-agent", user_agent.clone())
                .header("originator", "codex_cli_rs")
                .timeout(timeout);

            request_builder = match route.auth_header.as_deref() {
                Some("x-api-key") => request_builder.header("x-api-key", &route.api_key),
                _ => request_builder.header("authorization", format!("Bearer {}", route.api_key)),
            };
            if let Some(headers) = &route.headers {
                for (key, value) in headers {
                    if let Some(v) = value.as_str() {
                        request_builder = request_builder.header(key.as_str(), v);
                    }
                }
            }

            let response = request_builder
                .json(&body)
                .send()
                .await
                .map_err(Self::map_request_error)?;
            let status = response.status().as_u16();

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                let error = Self::http_status_error(status, error_text);
                if index + 1 < urls.len() && matches!(status, 404 | 405 | 502 | 503) {
                    last_error = Some(error);
                    continue;
                }
                return Err(error);
            }

            // 完整读取流式响应，直到正常结束或错误
            match Self::consume_chat_stream_full(response.bytes_stream()).await {
                Ok(()) => {
                    log::debug!(
                        "[StreamCheck][Devin] Chat stream completed normally for {}",
                        url
                    );
                    return Ok((status, actual_model));
                }
                Err(e) => {
                    log::warn!("[StreamCheck][Devin] Chat stream error for {}: {}", url, e);
                    if index + 1 < urls.len() {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            AppError::Message("No valid Devin chat completions endpoint found".to_string())
        }))
    }

    async fn check_devin_responses_stream(
        client: &Client,
        provider: &Provider,
        route: &DevinCatalogRoute,
        test_prompt: &str,
        timeout: std::time::Duration,
    ) -> Result<(u16, String), AppError> {
        let url = Self::build_joycode_color_gateway_url("anthropic_completions");
        let mut body = if route.responses_mode.as_deref() == Some("codex") {
            json!({
                "model": route.upstream_model,
                "input": [{ "role": "user", "content": test_prompt }],
                "stream": true,
                "store": false,
                "max_output_tokens": 10
            })
        } else {
            json!({
                "model": route.upstream_model,
                "input": [{ "role": "user", "content": test_prompt }],
                "stream": true,
                "max_output_tokens": 10
            })
        };

        if route.responses_fast_mode {
            body["service_tier"] = json!("priority");
        }

        let mut request_builder = client
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("accept-encoding", "identity")
            .header("user-agent", "codex_cli_rs/0.80.0")
            .header("originator", "codex_cli_rs")
            .timeout(timeout);

        request_builder = match route.auth_header.as_deref() {
            Some("x-api-key") => request_builder.header("x-api-key", &route.api_key),
            _ => request_builder.header("authorization", format!("Bearer {}", route.api_key)),
        };
        if let Some(headers) = &route.headers {
            for (key, value) in headers {
                if let Some(v) = value.as_str() {
                    request_builder = request_builder.header(key.as_str(), v);
                }
            }
        }
        if let Some(ua) = Self::custom_user_agent(provider) {
            request_builder = request_builder.header("user-agent", ua);
        }

        let response = request_builder
            .json(&body)
            .send()
            .await
            .map_err(Self::map_request_error)?;
        let status = response.status().as_u16();
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::http_status_error(status, error_text));
        }

        // 完整读取流式响应，直到正常结束或错误
        Self::consume_responses_stream_full(response.bytes_stream()).await?;
        log::debug!(
            "[StreamCheck][Devin] Responses stream completed normally for {}",
            url
        );
        Ok((status, route.upstream_model.clone()))
    }

    async fn check_devin_route_once(
        client: &Client,
        provider: &Provider,
        route: &DevinCatalogRoute,
        test_prompt: &str,
        timeout: std::time::Duration,
    ) -> Result<(u16, String), AppError> {
        let is_joycode =
            route.base_url.contains("joycode-api.jd.com") || route.base_url.contains("joycode");

        if is_joycode && route.endpoint.contains("/api/saas/anthropic/v1/messages") {
            return Self::check_joycode_anthropic_stream(client, route, test_prompt, timeout).await;
        }

        let auth = AuthInfo::new(route.api_key.clone(), route.auth_strategy());

        match route.endpoint.as_str() {
            "/v1/messages" | "/messages" | "/api/saas/anthropic/v1/messages" => {
                Self::check_claude_stream(
                    client,
                    &route.base_url,
                    &auth,
                    &route.upstream_model,
                    test_prompt,
                    timeout,
                    provider,
                    Some("anthropic"),
                    route.headers.as_ref(),
                )
                .await
            }
            "/v1/chat/completions"
            | "/chat/completions"
            | "/api/saas/openai/v2/chat/completions" => {
                Self::check_devin_chat_stream(client, provider, route, test_prompt, timeout).await
            }
            "/v1/responses"
            | "/responses"
            | "/v1/responses/compact"
            | "/responses/compact"
            | "/api/saas/openai/v1/responses" => {
                Self::check_devin_responses_stream(client, provider, route, test_prompt, timeout)
                    .await
            }
            other => Err(AppError::Message(format!(
                "Unsupported Devin route endpoint: {other}"
            ))),
        }
    }

    async fn check_joycode_anthropic_stream(
        _client: &Client,
        route: &DevinCatalogRoute,
        test_prompt: &str,
        timeout: std::time::Duration,
    ) -> Result<(u16, String), AppError> {
        let pt_key = route
            .headers
            .as_ref()
            .and_then(|headers| Self::find_header_value_case_insensitive(headers, "ptKey"))
            .or_else(|| {
                route
                    .headers
                    .as_ref()
                    .and_then(|headers| Self::find_header_value_case_insensitive(headers, "Cookie"))
                    .and_then(|cookie| Self::cookie_value(&cookie, "pt_key"))
            })
            .ok_or_else(|| {
                AppError::Message(
                    "JoyCode API test requires ptKey or joycode_cookie in Devin settings"
                        .to_string(),
                )
            })?;
        let login_type = route
            .headers
            .as_ref()
            .and_then(|headers| Self::find_header_value_case_insensitive(headers, "loginType"))
            .unwrap_or_else(|| "ERP".to_string());
        let tenant = route
            .headers
            .as_ref()
            .and_then(|headers| Self::find_header_value_case_insensitive(headers, "tenant"))
            .unwrap_or_else(|| "JD".to_string());

        let adapter = crate::proxy::providers::JoyCodeAnthropicAdapter::new();
        let prepared = adapter
            .get_model_prepare(
                &route.base_url,
                &route.upstream_model,
                true,
                &pt_key,
                &login_type,
                &tenant,
                "京东集团",
            )
            .await
            .map_err(|e| AppError::Message(format!("JoyCode prepare failed: {e}")))?;

        let endpoint = route.endpoint.trim_start_matches('/');
        let url = format!("{}/{}", route.base_url.trim_end_matches('/'), endpoint);
        let body = json!({
            "model": route.upstream_model,
            "max_tokens": 32,
            "messages": [{ "role": "user", "content": test_prompt }],
            "stream": true,
            "chatId": prepared.chat_id,
            "tenant": tenant,
            "orgFullName": "京东集团",
            "userId": "",
            "client": crate::proxy::providers::JOYCODE_VSCODE_CLIENT,
            "clientVersion": crate::proxy::providers::JOYCODE_VSCODE_CLIENT_VERSION,
            "language": crate::proxy::providers::JOYCODE_DEFAULT_LANGUAGE
        });

        let client = crate::proxy::http_client::get_direct();
        let mut request_builder = client
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("accept-encoding", "identity")
            .header("user-agent", "VS Code/3.8.58")
            .header("x-model-token", prepared.token)
            .header("x-ms-client-request-id", uuid::Uuid::new_v4().to_string())
            .header("ptkey", &pt_key)
            .header("logintype", &login_type)
            .header("tenant", &tenant)
            .timeout(timeout);

        if let Some(headers) = &route.headers {
            for (key, value) in headers {
                if key.eq_ignore_ascii_case("cookie")
                    || key.eq_ignore_ascii_case("ptKey")
                    || key.eq_ignore_ascii_case("ptkey")
                    || key.eq_ignore_ascii_case("pt_key")
                    || key.eq_ignore_ascii_case("x-pt-key")
                    || key.eq_ignore_ascii_case("loginType")
                    || key.eq_ignore_ascii_case("tenant")
                    || key.eq_ignore_ascii_case("anthropic-version")
                    || key.eq_ignore_ascii_case("anthropic-beta")
                    || key.eq_ignore_ascii_case("authorization")
                    || key.eq_ignore_ascii_case("x-api-key")
                {
                    continue;
                }
                if let Some(v) = value.as_str() {
                    request_builder = request_builder.header(key.as_str(), v);
                }
            }
        }

        let response = request_builder
            .json(&body)
            .send()
            .await
            .map_err(Self::map_request_error)?;
        let status = response.status().as_u16();
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::http_status_error(status, error_text));
        }

        Self::consume_anthropic_stream_full(response.bytes_stream()).await?;
        log::debug!(
            "[StreamCheck][Devin] JoyCode Anthropic stream completed normally for {}",
            url
        );
        Ok((status, route.upstream_model.clone()))
    }

    pub(crate) fn classify_http_status(status: u16) -> &'static str {
        match status {
            400 => "Bad request (400)",
            401 => "Auth rejected (401)",
            402 => "Payment required (402)",
            403 => "Access denied (403)",
            404 => "Not found (404)",
            429 => "Rate limited (429)",
            500 => "Internal server error (500)",
            502 => "Bad gateway (502)",
            503 => "Service unavailable (503)",
            504 => "Gateway timeout (504)",
            s if (500..600).contains(&s) => "Server error",
            _ => "HTTP error",
        }
    }

    async fn consume_anthropic_stream_full(
        mut stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    ) -> Result<(), AppError> {
        let mut buffer = String::new();
        let mut current_event: Option<String> = None;
        let mut saw_delta = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| AppError::Message(format!("Stream read error: {e}")))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.is_empty() {
                    current_event = None;
                    continue;
                }

                let line = Self::unwrap_nested_sse_line(&line);

                if let Some(event_type) = line.strip_prefix("event:").map(str::trim) {
                    match event_type {
                        "message_stop" => return Ok(()),
                        "error" => current_event = Some("error".to_string()),
                        other => current_event = Some(other.to_string()),
                    }
                    continue;
                }

                let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                let data = Self::unwrap_nested_sse_data(data);
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    return Ok(());
                }

                let Ok(json) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };

                if json.get("error").is_some() || current_event.as_deref() == Some("error") {
                    return Err(AppError::Message(format!(
                        "Anthropic stream error: {}",
                        Self::stream_error_message(&json)
                    )));
                }

                if json
                    .get("code")
                    .and_then(Value::as_i64)
                    .is_some_and(|code| code != 0)
                {
                    return Err(AppError::Message(format!(
                        "Anthropic stream business error: {}",
                        Self::stream_error_message(&json)
                    )));
                }

                match json.get("type").and_then(Value::as_str) {
                    Some("message_stop") => return Ok(()),
                    Some("error") => {
                        return Err(AppError::Message(format!(
                            "Anthropic stream error: {}",
                            Self::stream_error_message(&json)
                        )));
                    }
                    Some("content_block_delta" | "message_delta") => {
                        saw_delta = true;
                    }
                    _ => {}
                }
            }
        }

        if let Ok(json) = serde_json::from_str::<Value>(buffer.trim()) {
            if json.get("error").is_some()
                || json
                    .get("code")
                    .and_then(Value::as_i64)
                    .is_some_and(|code| code != 0)
            {
                return Err(AppError::Message(format!(
                    "Anthropic stream error: {}",
                    Self::stream_error_message(&json)
                )));
            }
        }

        if saw_delta {
            return Err(AppError::Message(
                "Stream ended before completion after receiving partial Anthropic data".to_string(),
            ));
        }

        Err(AppError::Message(
            "Stream ended without Anthropic completion signal".to_string(),
        ))
    }

    async fn consume_chat_stream_full(
        mut stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    ) -> Result<(), AppError> {
        let mut buffer = String::new();
        let mut saw_content = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| AppError::Message(format!("Stream read error: {e}")))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // 逐行解析 SSE 事件
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.is_empty() {
                    continue;
                }

                let line = Self::unwrap_nested_sse_line(&line);

                // 检测 [DONE] 信号
                if line.contains("[DONE]") {
                    log::debug!("[StreamCheck] Chat stream received [DONE]");
                    return Ok(());
                }

                // 解析 data: 行
                if let Some(data) = line.strip_prefix("data:").map(str::trim) {
                    let data = Self::unwrap_nested_sse_data(data);
                    if data.is_empty() || data == "[DONE]" {
                        return Ok(());
                    }

                    if let Ok(json) = serde_json::from_str::<Value>(&data) {
                        // 检查错误
                        if json.get("error").is_some() {
                            let error_msg = json
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .unwrap_or("Unknown error in stream");
                            return Err(AppError::Message(format!("Stream error: {error_msg}")));
                        }

                        // 检查内容
                        if json.pointer("/choices/0/delta/content").is_some()
                            || json.pointer("/choices/0/delta/reasoning_content").is_some()
                        {
                            saw_content = true;
                        }

                        // 检查 finish_reason
                        if json.pointer("/choices/0/finish_reason").is_some() {
                            log::debug!("[StreamCheck] Chat stream received finish_reason");
                            return Ok(());
                        }
                    }
                }
            }
        }

        // 流结束但没收到完成信号
        if saw_content {
            // 收到过内容但流异常断开，视为部分成功（某些上游可能不发 [DONE]）
            log::warn!("[StreamCheck] Chat stream ended without [DONE] but saw content");
            Ok(())
        } else {
            Err(AppError::Message(
                "Stream ended without content or completion signal".to_string(),
            ))
        }
    }

    async fn consume_responses_stream_full(
        mut stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    ) -> Result<(), AppError> {
        let mut buffer = String::new();
        let mut saw_content = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| AppError::Message(format!("Stream read error: {e}")))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // 逐行解析 SSE 事件
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.is_empty() {
                    continue;
                }

                let line = Self::unwrap_nested_sse_line(&line);

                // 解析 event: 和 data: 行
                if let Some(event_type) = line.strip_prefix("event:").map(str::trim) {
                    match event_type {
                        "response.completed" | "response.done" => {
                            log::debug!("[StreamCheck] Responses stream received completion event");
                            return Ok(());
                        }
                        "response.failed" | "response.error" => {
                            return Err(AppError::Message(format!(
                                "Responses stream failed: {event_type}"
                            )));
                        }
                        "response.output_text.delta" | "response.content_block.delta" => {
                            saw_content = true;
                        }
                        _ => {}
                    }
                }

                if let Some(data) = line.strip_prefix("data:").map(str::trim) {
                    let data = Self::unwrap_nested_sse_data(data);
                    if data.is_empty() {
                        continue;
                    }

                    if let Ok(json) = serde_json::from_str::<Value>(&data) {
                        // 检查 type 字段
                        match json.get("type").and_then(Value::as_str) {
                            Some("response.completed") | Some("response.done") => {
                                log::debug!("[StreamCheck] Responses stream completed");
                                return Ok(());
                            }
                            Some("response.failed") | Some("error") => {
                                let error_msg = json
                                    .pointer("/error/message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("Unknown error in responses stream");
                                return Err(AppError::Message(format!(
                                    "Responses stream error: {error_msg}"
                                )));
                            }
                            Some(
                                "response.output_text.delta"
                                | "response.content_block.delta"
                                | "content_block_delta",
                            ) => {
                                saw_content = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // 流结束但没收到完成信号
        if saw_content {
            // 收到过内容但流异常断开，视为部分成功（某些上游可能不发完成事件）
            log::warn!(
                "[StreamCheck] Responses stream ended without completion event but saw content"
            );
            Ok(())
        } else {
            Err(AppError::Message(
                "Responses stream ended without content or completion signal".to_string(),
            ))
        }
    }

    fn cookie_value(cookie: &str, name: &str) -> Option<String> {
        cookie.split(';').find_map(|part| {
            let (key, value) = part.trim().split_once('=')?;
            key.eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
    }

    pub(crate) fn detect_error_category(status: u16, body: &str) -> Option<&'static str> {
        // 只检查 4xx；5xx 的错误信息里可能巧合出现"model"之类的词，容易误判
        if !(400..500).contains(&status) {
            return None;
        }
        let lower = body.to_lowercase();
        let qianfan_quota_indicators = [
            "coding_plan_hour_quota_exceeded",
            "coding_plan_week_quota_exceeded",
            "coding_plan_month_quota_exceeded",
        ];
        if qianfan_quota_indicators.iter().any(|s| lower.contains(s)) {
            return Some("quotaExceeded");
        }

        // 必须提到 "model"，避免通用 404 / 400 被误判
        if !lower.contains("model") {
            return None;
        }
        let indicators = [
            "model_not_found",
            "model not found",
            "does not exist",
            "invalid_model",
            "invalid model",
            "unknown_model",
            "unknown model",
            "is not a valid model",
            "not_found_error", // Anthropic 的 type 字段
        ];
        if indicators.iter().any(|s| lower.contains(s)) {
            return Some("modelNotFound");
        }
        None
    }

    fn devin_catalog_route_from_model(model: &Value) -> Option<DevinCatalogRoute> {
        let route = model
            .get("routes")
            .and_then(Value::as_array)
            .and_then(|routes| {
                routes
                    .iter()
                    .filter(|route| {
                        route
                            .get("enabled")
                            .and_then(Value::as_bool)
                            .unwrap_or(true)
                    })
                    .min_by_key(|route| {
                        route.get("priority").and_then(Value::as_i64).unwrap_or(100)
                    })
            });

        let request_model = string_field(model, &["model"])?;
        let upstream_model = string_field(model, &["upstreamModel", "upstream_model"])
            .unwrap_or_else(|| request_model.clone());
        let endpoint = string_field(model, &["endpoint", "path"])
            .or_else(|| route.and_then(|route| string_field(route, &["endpoint", "path"])))?;
        let base_url = string_field(model, &["baseUrl", "base_url", "host", "url"])
            .or_else(|| {
                route.and_then(|route| string_field(route, &["baseUrl", "base_url", "host", "url"]))
            })?
            .trim_end_matches('/')
            .to_string();
        let api_key = string_field(model, &["apiKey", "api_key", "key"]).or_else(|| {
            route.and_then(|route| string_field(route, &["apiKey", "api_key", "key"]))
        })?;
        let auth_header = string_field(model, &["authHeader", "auth_header"])
            .or_else(|| route.and_then(|route| string_field(route, &["authHeader", "auth_header"])))
            .map(|value| value.to_ascii_lowercase());
        let mut headers = header_fields(model, &["headers", "extraHeaders", "extra_headers"]);
        if let Some(route) = route {
            merge_header_fields(
                &mut headers,
                header_fields(route, &["headers", "extraHeaders", "extra_headers"]),
            );
        }
        let headers = (!headers.is_empty()).then_some(headers);
        let responses_mode =
            string_field(model, &["responsesMode", "responses_mode"]).or_else(|| {
                route.and_then(|route| string_field(route, &["responsesMode", "responses_mode"]))
            });
        let responses_fast_mode = bool_field(model, &["responsesFastMode", "responses_fast_mode"])
            .or_else(|| {
                route.and_then(|route| {
                    bool_field(route, &["responsesFastMode", "responses_fast_mode"])
                })
            })
            .unwrap_or(false);

        Some(DevinCatalogRoute {
            request_model,
            upstream_model,
            endpoint: Self::normalize_devin_endpoint(&endpoint),
            base_url,
            api_key,
            auth_header,
            headers,
            responses_mode,
            responses_fast_mode,
        })
    }

    fn devin_catalog_routes(provider: &Provider) -> Vec<DevinCatalogRoute> {
        let Some(models) = provider
            .settings_config
            .pointer("/modelCatalog/models")
            .and_then(Value::as_array)
        else {
            return Vec::new();
        };

        models
            .iter()
            .filter_map(Self::devin_catalog_route_from_model)
            .collect()
    }

    fn extract_codex_model(provider: &Provider) -> Option<String> {
        let config_text = provider
            .settings_config
            .get("config")
            .and_then(|value| value.as_str())?;
        if config_text.trim().is_empty() {
            return None;
        }

        let table = toml::from_str::<toml::Table>(config_text).ok()?;
        table
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn extract_env_model(provider: &Provider, key: &str) -> Option<String> {
        provider
            .settings_config
            .get("env")
            .and_then(|env| env.get(key))
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn extract_openclaw_model(provider: &Provider) -> Option<String> {
        // OpenClaw uses models array: [{ "id": "model-id", "name": "Model Name" }]
        let models = provider
            .settings_config
            .get("models")
            .and_then(|m| m.as_array())?;

        // Return the first model ID from the models array
        models
            .first()
            .and_then(|m| m.get("id"))
            .and_then(|id| id.as_str())
            .map(|s| s.to_string())
    }

    fn extract_opencode_model(provider: &Provider) -> Option<String> {
        let models = provider
            .settings_config
            .get("models")
            .and_then(|m| m.as_object())?;

        // Return the first model ID from the models map
        models.keys().next().map(|s| s.to_string())
    }

    fn find_header_value_case_insensitive(
        headers: &serde_json::Map<String, Value>,
        name: &str,
    ) -> Option<String> {
        headers.iter().find_map(|(key, value)| {
            key.eq_ignore_ascii_case(name)
                .then(|| value.as_str().map(str::to_string))
                .flatten()
        })
    }

    fn get_arch_name() -> &'static str {
        match std::env::consts::ARCH {
            "aarch64" => "arm64",
            "x86_64" => "x86_64",
            "x86" => "x86",
            other => other,
        }
    }

    fn get_os_name() -> &'static str {
        match std::env::consts::OS {
            "macos" => "MacOS",
            "linux" => "Linux",
            "windows" => "Windows",
            other => other,
        }
    }

    fn http_status_error(status: u16, body: String) -> AppError {
        let body = if body.len() > 200 {
            // 安全截断：找到 200 字节内最近的 char 边界
            let mut end = 200;
            while end > 0 && !body.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &body[..end])
        } else {
            body
        };
        AppError::HttpStatus { status, body }
    }

    fn joycode_color_gateway_sign(params: &[(&str, String)]) -> String {
        type HmacSha256 = Hmac<sha2::Sha256>;
        const JOYCODE_COLOR_GATEWAY_SECRET: &[u8] = b"0691a3f0b37b4a85aeb63ad0fc7db3ed";

        let mut pairs = params.iter().collect::<Vec<_>>();
        pairs.sort_by(|left, right| left.0.cmp(right.0));
        let sign_text = pairs
            .into_iter()
            .map(|(_, value)| value.as_str())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join("&");

        let mut mac = HmacSha256::new_from_slice(JOYCODE_COLOR_GATEWAY_SECRET)
            .expect("JoyCode color gateway HMAC key is valid");
        mac.update(sign_text.as_bytes());
        mac.finalize()
            .into_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn normalize_devin_endpoint(endpoint: &str) -> String {
        let raw = endpoint.trim();
        let with_slash = if raw.starts_with('/') {
            raw.to_string()
        } else {
            format!("/{raw}")
        };
        match with_slash.as_str() {
            "/messages" => "/v1/messages".to_string(),
            "/responses" => "/v1/responses".to_string(),
            "/responses/compact" => "/v1/responses/compact".to_string(),
            "/chat/completions" => "/v1/chat/completions".to_string(),
            _ => with_slash,
        }
    }

    fn parse_model_with_effort(model: &str) -> (String, Option<String>) {
        if let Some(pos) = model.find('@').or_else(|| model.find('#')) {
            let actual_model = model[..pos].to_string();
            let effort = model[pos + 1..].to_string();
            if !effort.is_empty() {
                return (actual_model, Some(effort));
            }
        }
        (model.to_string(), None)
    }

    fn resolve_claude_stream_url(
        base_url: &str,
        auth_strategy: AuthStrategy,
        api_format: &str,
        is_full_url: bool,
        model: &str,
    ) -> String {
        if api_format == "gemini_native" {
            // Strip an optional `models/` resource-name prefix so that model
            // identifiers copied from Gemini SDK outputs (e.g.
            // `models/gemini-2.5-pro`) don't produce a doubled
            // `/v1beta/models/models/...` URL.
            let normalized_model = normalize_gemini_model_id(model);
            let endpoint =
                format!("/v1beta/models/{normalized_model}:streamGenerateContent?alt=sse");
            return resolve_gemini_native_url(base_url, &endpoint, is_full_url);
        }

        if is_full_url {
            return base_url.to_string();
        }

        let base = base_url.trim_end_matches('/');
        let is_github_copilot = auth_strategy == AuthStrategy::GitHubCopilot;

        if is_github_copilot && api_format == "openai_responses" {
            format!("{base}/v1/responses")
        } else if is_github_copilot {
            format!("{base}/chat/completions")
        } else if api_format == "openai_responses" {
            if base.ends_with("/v1") {
                format!("{base}/responses")
            } else {
                format!("{base}/v1/responses")
            }
        } else if api_format == "openai_chat" {
            if base.ends_with("/v1") {
                format!("{base}/chat/completions")
            } else {
                format!("{base}/v1/chat/completions")
            }
        } else if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        }
    }

    fn resolve_codex_chat_stream_urls(base_url: &str, is_full_url: bool) -> Vec<String> {
        Self::resolve_codex_endpoint_urls(base_url, is_full_url, "chat/completions")
    }

    fn resolve_codex_endpoint_urls(
        base_url: &str,
        is_full_url: bool,
        endpoint: &str,
    ) -> Vec<String> {
        if is_full_url {
            return vec![base_url.to_string()];
        }

        let base = base_url.trim_end_matches('/');
        let lower = base.to_ascii_lowercase();
        let endpoint_suffix = format!("/{endpoint}");
        let endpoint_lower = endpoint_suffix.to_ascii_lowercase();

        // 用户在 base_url 里写了完整 endpoint 但忘开 is_full_url 的兜底
        if lower.ends_with(&endpoint_lower) {
            return vec![base.to_string()];
        }

        if base.ends_with("/v1") {
            return vec![format!("{base}{endpoint_suffix}")];
        }

        if crate::proxy::providers::is_origin_only_url(base) {
            vec![
                format!("{base}/v1{endpoint_suffix}"),
                format!("{base}{endpoint_suffix}"),
            ]
        } else {
            vec![
                format!("{base}{endpoint_suffix}"),
                format!("{base}/v1{endpoint_suffix}"),
            ]
        }
    }

    fn resolve_codex_stream_urls(base_url: &str, is_full_url: bool) -> Vec<String> {
        Self::resolve_codex_endpoint_urls(base_url, is_full_url, "responses")
    }

    fn resolve_test_model(
        app_type: &AppType,
        provider: &Provider,
        config: &StreamCheckConfig,
    ) -> String {
        match app_type {
            AppType::Claude | AppType::ClaudeDesktop => {
                Self::extract_env_model(provider, "ANTHROPIC_MODEL")
                    .unwrap_or_else(|| config.claude_model.clone())
            }
            AppType::Codex | AppType::Devin => {
                Self::extract_codex_model(provider).unwrap_or_else(|| config.codex_model.clone())
            }
            AppType::Gemini => Self::extract_env_model(provider, "GEMINI_MODEL")
                .unwrap_or_else(|| config.gemini_model.clone()),
            AppType::OpenCode => {
                // OpenCode uses models map in settings_config
                // Try to extract first model from the models object
                Self::extract_opencode_model(provider).unwrap_or_else(|| "gpt-4o".to_string())
            }
            AppType::OpenClaw | AppType::Hermes => {
                // OpenClaw/Hermes use models array in settings_config
                // Try to extract first model from the models array
                Self::extract_openclaw_model(provider).unwrap_or_else(|| "gpt-4o".to_string())
            }
        }
    }

    fn stream_error_message(json: &Value) -> String {
        json.pointer("/error/message")
            .or_else(|| json.get("message"))
            .or_else(|| json.get("msg"))
            .and_then(Value::as_str)
            .unwrap_or("unknown upstream error")
            .chars()
            .take(200)
            .collect()
    }

    pub fn summarize_results(results: &[StreamCheckResult]) -> StreamCheckResult {
        let tested_at = chrono::Utc::now().timestamp();
        if results.is_empty() {
            return StreamCheckResult {
                status: HealthStatus::Failed,
                success: false,
                message: "No models tested".to_string(),
                response_time_ms: None,
                http_status: None,
                model_used: String::new(),
                tested_at,
                retry_count: 0,
                error_category: None,
            };
        }

        let success_count = results.iter().filter(|r| r.success).count();
        let total = results.len();
       let status = if success_count == total {
           if results.iter().any(|r| r.status == HealthStatus::Degraded) {
               HealthStatus::Degraded
           } else {
               HealthStatus::Operational
           }
       } else {
            // Partial model failures (e.g. small models mapped to upstream
            // models the provider doesn't support) should degrade, not fail.
            if success_count > 0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Failed
            }
       };
        let failed = results
            .iter()
            .filter(|r| !r.success)
            .map(|r| format!("{}: {}", r.model_used, r.message))
            .collect::<Vec<_>>();

       StreamCheckResult {
           status,
            // Provider is usable as long as at least one model passes.
            success: success_count > 0,
           message: if failed.is_empty() {
                format!("All {total} models check succeeded")
            } else {
                format!(
                    "{success_count}/{total} models succeeded; {}",
                    failed.join("; ")
                )
            },
            response_time_ms: results.iter().filter_map(|r| r.response_time_ms).max(),
            http_status: results
                .iter()
                .find_map(|r| (!r.success).then_some(r.http_status).flatten()),
            model_used: format!("{total} models"),
            tested_at,
            retry_count: results.iter().map(|r| r.retry_count).max().unwrap_or(0),
            error_category: results.iter().find_map(|r| r.error_category.clone()),
        }
    }

    fn unwrap_nested_sse_data(data: &str) -> String {
        let mut current = data.trim();
        while let Some(inner) = current.strip_prefix("data:").map(str::trim) {
            current = inner;
        }
        current.to_string()
    }

    fn unwrap_nested_sse_line(line: &str) -> String {
        let mut current = line.trim();
        while let Some(inner) = current.strip_prefix("data:").map(str::trim) {
            if inner.starts_with("event:") || inner.starts_with("data:") {
                current = inner;
            } else {
                break;
            }
        }
        current.to_string()
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(settings_config: serde_json::Value) -> Provider {
        Provider::with_id(
            "test".to_string(),
            "Test".to_string(),
            settings_config,
            None,
        )
    }

    #[test]
    fn test_default_config_uses_reachability_friendly_values() {
        let config = StreamCheckConfig::default();
        assert_eq!(config.timeout_secs, 8);
        assert_eq!(config.max_retries, 1);
        // 降级阈值沿用旧尺度，避免把 1 秒多的正常延迟误判为"较慢"
        assert_eq!(config.degraded_threshold_ms, 6000);
    }

    #[test]
    fn test_determine_status() {
        assert_eq!(
            StreamCheckService::determine_status(1000, 1500),
            HealthStatus::Operational
        );
        assert_eq!(
            StreamCheckService::determine_status(1500, 1500),
            HealthStatus::Operational
        );
        assert_eq!(
            StreamCheckService::determine_status(1501, 1500),
            HealthStatus::Degraded
        );
    }

    #[test]
    fn test_should_retry_only_on_timeout_like_errors() {
        assert!(StreamCheckService::should_retry("Request timeout"));
        assert!(StreamCheckService::should_retry("request timed out"));
        assert!(StreamCheckService::should_retry("connection abort"));
        // 连接被拒 / DNS 失败不重试
        assert!(!StreamCheckService::should_retry(
            "Connection failed: dns error"
        ));
        assert!(!StreamCheckService::should_retry("Reachable"));
    }

    #[test]
    fn test_build_result_any_http_status_is_reachable() {
        // 任何 HTTP 状态码都算可达（success=true）
        for status in [200u16, 401, 403, 404, 429, 500, 503] {
            let r = StreamCheckService::build_result(Ok(status), 100, 1500);
            assert!(r.success, "status {status} should be reachable");
            assert_eq!(r.status, HealthStatus::Operational);
            assert_eq!(r.http_status, Some(status));
            assert!(r.model_used.is_empty());
            assert!(r.error_category.is_none());
        }
    }

    #[test]
    fn test_build_result_network_error_is_unreachable() {
        let r = StreamCheckService::build_result(
            Err(AppError::Message("Connection failed: refused".to_string())),
            5,
            1500,
        );
        assert!(!r.success);
        assert_eq!(r.status, HealthStatus::Failed);
        assert!(r.http_status.is_none());
    }

    #[test]
    fn test_build_result_slow_response_is_degraded() {
        let r = StreamCheckService::build_result(Ok(200), 3000, 1500);
        assert!(r.success);
        assert_eq!(r.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_merge_provider_config_override_and_default() {
        use crate::provider::{ProviderMeta, ProviderTestConfig};

        let global = StreamCheckConfig::default();

        // 无 testConfig → 用全局
        let p = make_provider(serde_json::json!({}));
        let merged = StreamCheckService::merge_provider_config(&p, &global);
        assert_eq!(merged.timeout_secs, global.timeout_secs);

        // testConfig 启用并覆盖部分字段
        let mut p2 = make_provider(serde_json::json!({}));
       p2.meta = Some(ProviderMeta {
           test_config: Some(ProviderTestConfig {
               enabled: true,
               timeout_secs: Some(20),
               degraded_threshold_ms: Some(3000),
               max_retries: None,
               test_prompt: None,
               test_model: None,
           }),
           ..Default::default()
       });
        let merged2 = StreamCheckService::merge_provider_config(&p2, &global);
        assert_eq!(merged2.timeout_secs, 20);
        assert_eq!(merged2.degraded_threshold_ms, 3000);
        assert_eq!(merged2.max_retries, global.max_retries); // 未覆盖 → 全局

        // testConfig 存在但未启用 → 忽略，用全局
        let mut p3 = make_provider(serde_json::json!({}));
       p3.meta = Some(ProviderMeta {
           test_config: Some(ProviderTestConfig {
               enabled: false,
               timeout_secs: Some(99),
               degraded_threshold_ms: None,
               max_retries: None,
               test_prompt: None,
               test_model: None,
           }),
           ..Default::default()
       });
        let merged3 = StreamCheckService::merge_provider_config(&p3, &global);
        assert_eq!(merged3.timeout_secs, global.timeout_secs);
    }

    #[test]
    fn test_resolve_opencode_base_url_explicit_wins() {
        let p = make_provider(serde_json::json!({
            "npm": "@ai-sdk/openai",
            "options": { "baseURL": "https://proxy.local/v1", "apiKey": "k" },
            "models": {},
        }));
        let resolved =
            StreamCheckService::resolve_opencode_base_url(&p, Some("@ai-sdk/openai")).unwrap();
        assert_eq!(resolved, "https://proxy.local/v1");
    }

    #[test]
    fn test_resolve_opencode_base_url_falls_back_for_known_npm() {
        let p = make_provider(serde_json::json!({
            "npm": "@ai-sdk/anthropic",
            "options": { "apiKey": "k" },
            "models": {},
        }));
        let resolved =
            StreamCheckService::resolve_opencode_base_url(&p, Some("@ai-sdk/anthropic")).unwrap();
        assert_eq!(resolved, "https://api.anthropic.com");
    }

    #[test]
    fn test_resolve_opencode_base_url_errors_for_openai_compatible_without_url() {
        let p = make_provider(serde_json::json!({
            "npm": "@ai-sdk/openai-compatible",
            "options": { "apiKey": "k" },
            "models": {},
        }));
        let result =
            StreamCheckService::resolve_opencode_base_url(&p, Some("@ai-sdk/openai-compatible"));
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_openclaw_base_url_missing_errors() {
        let p = make_provider(serde_json::json!({ "apiKey": "k", "api": "openai-completions" }));
        assert!(StreamCheckService::extract_openclaw_base_url(&p).is_err());

        let p2 = make_provider(serde_json::json!({ "baseUrl": "https://api.deepseek.com/v1" }));
        assert_eq!(
            StreamCheckService::extract_openclaw_base_url(&p2).unwrap(),
            "https://api.deepseek.com/v1"
        );
    }

    #[test]
    fn test_resolve_base_url_uses_explicit_url_or_errors_when_missing() {
        // 有显式 base_url → 直接用
        let p = make_provider(
            serde_json::json!({ "env": { "ANTHROPIC_BASE_URL": "https://relay.example/v1" } }),
        );
        assert_eq!(
            StreamCheckService::resolve_base_url(&AppType::Claude, &p).unwrap(),
            "https://relay.example/v1"
        );

        // 缺 base_url（官方留空 / 用户忘填）→ 报错。官方供应商的检测按钮在前端已隐藏，
        // 不会走到这里；不做官方端点回退（避免给忘填地址的第三方误显绿灯）。
        let empty = make_provider(serde_json::json!({ "env": {} }));
        assert!(StreamCheckService::resolve_base_url(&AppType::Claude, &empty).is_err());
    }
}

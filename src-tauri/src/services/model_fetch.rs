//! 模型列表获取服务
//!
//! 通过 OpenAI 兼容的 GET /v1/models 端点获取供应商可用模型列表。
//! 主要面向第三方聚合站（硅基流动、OpenRouter 等），以及把 Anthropic
//! 协议挂在兼容子路径上的官方供应商（DeepSeek、Kimi、智谱 GLM 等）。

use hmac::{Hmac, Mac};
use reqwest::header::{HeaderValue, CONTENT_TYPE, USER_AGENT};
use reqwest::StatusCode;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;

/// 获取到的模型信息
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FetchedModel {
    pub id: String,
    pub owned_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responses_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JoyCodeLoginState {
    pub user_name: Option<String>,
    pub tenant: String,
    pub login_type: String,
    pub pt_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org_full_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// OpenAI 兼容的 /v1/models 响应格式
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Option<Vec<ModelEntry>>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    owned_by: Option<String>,
}

const FETCH_TIMEOUT_SECS: u64 = 15;
const JOYCODE_COLOR_GATEWAY_URL: &str = "https://api-ai.jd.com/api";
const JOYCODE_COLOR_GATEWAY_APPID: &str = "joycode_ide";
const JOYCODE_COLOR_GATEWAY_SECRET: &[u8] = b"0691a3f0b37b4a85aeb63ad0fc7db3ed";
const JOYCODE_PLUGIN_CONFIG_FUNCTION_ID: &str = "plugin_config";
const JOYCODE_MODEL_LIST_FUNCTION_ID: &str = "joycode_modelList";

/// 404/405 响应体截断长度：避免把几十 KB HTML 404 页整页保留到错误串里。
const ERROR_BODY_MAX_CHARS: usize = 512;

/// 已知的「Anthropic 协议兼容子路径」后缀；按长度降序，最长前缀优先匹配。
/// baseURL 命中这些后缀时，候选列表会追加「剥离后缀再拼 /v1/models / /models」的版本。
const KNOWN_COMPAT_SUFFIXES: &[&str] = &[
    "/api/claudecode",
    "/api/anthropic",
    "/apps/anthropic",
    "/api/coding",
    "/claudecode",
    "/anthropic",
    "/step_plan",
    "/coding",
    "/claude",
];

/// 获取供应商的可用模型列表
///
/// 使用 OpenAI 兼容的 GET /v1/models 端点，按候选列表顺序尝试。
pub async fn fetch_models(
    base_url: &str,
    api_key: &str,
    is_full_url: bool,
    models_url_override: Option<&str>,
    user_agent: Option<HeaderValue>,
) -> Result<Vec<FetchedModel>, String> {
    if api_key.is_empty() {
        return Err("API Key is required to fetch models".to_string());
    }

    let candidates = build_models_url_candidates(base_url, is_full_url, models_url_override)?;
    let client = crate::proxy::http_client::get();
    let mut last_err: Option<String> = None;

    for url in &candidates {
        log::debug!("[ModelFetch] Trying endpoint: {url}");
        let mut request = client
            .get(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS));
        // 自定义 User-Agent：部分 /models 端点同样有 UA 白名单（如 Kimi Coding Plan），
        // 与转发 / 检测路径共用同一 UA，避免"代理可用但取模型失败"。
        if let Some(ua) = &user_agent {
            request = request.header(USER_AGENT, ua.clone());
        }
        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                return Err(format!("Request failed: {e}"));
            }
        };

        let status = response.status();

        if status.is_success() {
            let resp: ModelsResponse = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {e}"))?;

            let mut models: Vec<FetchedModel> = resp
                .data
                .unwrap_or_default()
                .into_iter()
                .map(|m| FetchedModel {
                    id: m.id,
                    owned_by: m.owned_by,
                    display_name: None,
                    context_window: None,
                    upstream_model: None,
                    provider: None,
                    endpoint: None,
                    auth_header: None,
                    responses_mode: None,
                })
                .collect();

            models.sort_by(|a, b| a.id.cmp(&b.id));
            return Ok(models);
        }

        if status == StatusCode::NOT_FOUND || status == StatusCode::METHOD_NOT_ALLOWED {
            let body = truncate_body(response.text().await.unwrap_or_default());
            last_err = Some(format!("HTTP {status}: {body}"));
            continue;
        }

        let body = truncate_body(response.text().await.unwrap_or_default());
        return Err(format!("HTTP {status}: {body}"));
    }

    Err(format!(
        "All candidates failed: {}",
        last_err.unwrap_or_else(|| "no candidates".to_string())
    ))
}

/// 获取 JoyCode 官方模型列表。
///
/// 参考官方 JoyCode VS Code 插件：
/// 1. JD 租户优先请求 `plugin_config` 的 `customModelList`
/// 2. 无覆盖时请求 `joycode_modelList`
/// 3. 两个接口均通过 Color 网关签名，并带 JoyCode 登录态头
pub async fn fetch_joycode_models(
    auth_headers_json: Option<&str>,
) -> Result<Vec<FetchedModel>, String> {
    let auth = collect_joycode_auth(auth_headers_json)?;
    let client = crate::proxy::http_client::get();

    if auth.tenant == "JD" {
        if let Ok(Some(models)) = fetch_joycode_custom_model_list(&client, &auth).await {
            return Ok(models);
        }
    }

    let response =
        post_joycode_color_gateway(&client, JOYCODE_MODEL_LIST_FUNCTION_ID, &auth, json!({}))
            .await?;
    let models = normalize_joycode_model_list(response.get("data").unwrap_or(&response))?;
    Ok(models)
}

/// 从官方 JoyCode VS Code 插件的 globalState 中读取登录态。
pub fn read_vscode_joycode_login_state() -> Result<JoyCodeLoginState, String> {
    let state = read_vscode_joycode_global_state()?;
    let info = state
        .get("jdhLoginInfo")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "VS Code JoyCode 插件未找到 jdhLoginInfo，请先在 VS Code 中完成登录".to_string()
        })?;
    parse_joycode_login_state(&Value::Object(info.clone()))
}

async fn fetch_joycode_custom_model_list(
    client: &reqwest::Client,
    auth: &JoyCodeAuth,
) -> Result<Option<Vec<FetchedModel>>, String> {
    let response = post_joycode_color_gateway(
        client,
        JOYCODE_PLUGIN_CONFIG_FUNCTION_ID,
        auth,
        json!({ "sceneType": "customModelList" }),
    )
    .await?;
    let root = response.get("data").unwrap_or(&response);
    let Some(custom_model_list) = root.get("customModelList") else {
        return Ok(None);
    };
    let enabled = custom_model_list
        .get("enable")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !enabled {
        return Ok(None);
    }
    let Some(model_list) = custom_model_list.get("modelList") else {
        return Ok(None);
    };
    let models = normalize_joycode_model_list(model_list)?;
    if models.is_empty() {
        Ok(None)
    } else {
        Ok(Some(models))
    }
}

async fn post_joycode_color_gateway(
    client: &reqwest::Client,
    function_id: &str,
    auth: &JoyCodeAuth,
    body: Value,
) -> Result<Value, String> {
    let url = build_joycode_color_gateway_url(function_id)?;
    let response = client
        .post(url)
        .header(CONTENT_TYPE, "application/json; charset=UTF-8")
        .header("ptKey", auth.pt_key.clone())
        .header("loginType", auth.login_type.clone())
        .header("tenant", auth.tenant.clone())
        .json(&body)
        .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("JoyCode model list request failed: {e}"))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read JoyCode model list response: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", truncate_body(text)));
    }
    if text.contains("<!DOCTYPE html>") {
        return Err("Server returned HTML error page instead of JSON".to_string());
    }
    serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse JoyCode model list response: {e}"))
}

/// 构造「模型列表端点」的候选 URL 列表
///
/// 候选顺序：
/// 1. `models_url_override` 非空 → 只返回它
/// 2. baseURL 拼 `/v1/models`；若已以版本段 `/v{N}` 结尾（`/v1`、智谱
///    `/api/coding/paas/v4` 等），版本号已在路径里，改拼 `/models`
/// 3. 版本段非 `/v1`（如 `/v4`）时再追加 `/v1/models` 作为兜底次候选
/// 4. 若 baseURL 命中 [`KNOWN_COMPAT_SUFFIXES`]，剥离后缀再拼 `/v1/models`、`/models`
///
/// 结果已去重且保持首次出现顺序。
pub fn build_models_url_candidates(
    base_url: &str,
    is_full_url: bool,
    models_url_override: Option<&str>,
) -> Result<Vec<String>, String> {
    if let Some(raw) = models_url_override {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(vec![trimmed.to_string()]);
        }
    }

    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("Base URL is empty".to_string());
    }

    let mut candidates: Vec<String> = Vec::new();

    if is_full_url {
        if let Some(idx) = trimmed.find("/v1/") {
            candidates.push(format!("{}/v1/models", &trimmed[..idx]));
        } else if let Some(idx) = trimmed.rfind('/') {
            let root = &trimmed[..idx];
            if root.contains("://") && root.len() > root.find("://").unwrap() + 3 {
                candidates.push(format!("{root}/v1/models"));
            }
        }
        if candidates.is_empty() {
            return Err("Cannot derive models endpoint from full URL".to_string());
        }
        return Ok(candidates);
    }

    // baseURL 已以版本段 /v{N} 结尾时（如 `/v1`、智谱 `/api/coding/paas/v4`），
    // OpenAI 惯例的模型端点是 `{base}/models`，不能再补 `/v1`
    // （否则 .../coding/paas/v4/v1/models → 404）。
    if ends_with_version_segment(trimmed) {
        candidates.push(format!("{trimmed}/models"));
        // 版本段非 /v1 时，保留旧的 /v1/models 作为兜底次候选（正确路径已在前）。
        if !trimmed.ends_with("/v1") {
            candidates.push(format!("{trimmed}/v1/models"));
        }
    } else {
        candidates.push(format!("{trimmed}/v1/models"));
    }

    if let Some(stripped) = strip_compat_suffix(trimmed) {
        let root = stripped.trim_end_matches('/');
        if !root.is_empty() && root.contains("://") {
            candidates.push(format!("{root}/v1/models"));
            candidates.push(format!("{root}/models"));
        }
    }

    // 候选最多 3 条，线性去重即可，不值得上 HashSet。
    let mut unique: Vec<String> = Vec::with_capacity(candidates.len());
    for url in candidates {
        if !unique.iter().any(|u| u == &url) {
            unique.push(url);
        }
    }

    Ok(unique)
}

/// 截断响应体到 [`ERROR_BODY_MAX_CHARS`] 字符，避免 HTML 404 页占用错误串。
fn truncate_body(body: String) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        body
    } else {
        let mut s: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
        s.push('…');
        s
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JoyCodeAuth {
    pt_key: String,
    login_type: String,
    tenant: String,
}

fn build_joycode_color_gateway_url(function_id: &str) -> Result<String, String> {
    let timestamp_ms = chrono::Utc::now().timestamp_millis().to_string();
    let params = vec![
        ("appid", JOYCODE_COLOR_GATEWAY_APPID.to_string()),
        ("functionId", function_id.to_string()),
        ("t", timestamp_ms),
    ];
    let sign = joycode_color_gateway_sign(&params);
    let mut url = url::Url::parse(JOYCODE_COLOR_GATEWAY_URL)
        .map_err(|e| format!("Invalid JoyCode gateway URL: {e}"))?;
    {
        let mut query = url.query_pairs_mut();
        for (key, value) in &params {
            query.append_pair(key, value);
        }
        query.append_pair("sign", &sign);
    }
    Ok(url.to_string())
}

fn joycode_color_gateway_sign(params: &[(&str, String)]) -> String {
    type HmacSha256 = Hmac<sha2::Sha256>;

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

fn collect_joycode_auth(auth_headers_json: Option<&str>) -> Result<JoyCodeAuth, String> {
    let mut headers = Vec::new();

    if let Ok(cookie) = std::env::var("JOYCODE_COOKIE") {
        headers.push(("cookie".to_string(), cookie));
    }
    if let Ok(cookie) = std::env::var("DEVIN_JOYCODE_COOKIE") {
        headers.push(("cookie".to_string(), cookie));
    }

    if let Ok(home) = std::env::var("HOME") {
        for path in [
            format!("{home}/.ccswitch/joycode.env"),
            format!("{home}/.ccswitch/devin.toml"),
        ] {
            if let Ok(content) = std::fs::read_to_string(&path) {
                collect_joycode_headers_from_env_like(&content, &mut headers);
                collect_joycode_headers_from_toml(&content, &mut headers);
                break;
            }
        }
    }

    // 表单里显式填写的 Header 覆盖优先级最高；环境变量 / 文件只作为兜底。
    collect_joycode_headers_from_json_str(auth_headers_json, &mut headers);

    promote_joycode_cookie_header_pairs(&mut headers);

    let pt_key = find_header_value(&headers, &["ptKey", "ptkey", "pt_key", "x-pt-key"])
        .or_else(|| {
            find_header_value(&headers, &["cookie"])
                .and_then(|cookie| extract_cookie_value(&cookie, "pt_key"))
        })
        .ok_or_else(|| {
            "JoyCode 登录态缺少 ptKey；请在本地代理 Header 覆盖、JOYCODE_COOKIE 或 ~/.ccswitch/joycode.env 中配置".to_string()
        })?;

    let login_type = find_header_value(&headers, &["loginType", "logintype"])
        .unwrap_or_else(|| "ERP".to_string());
    let tenant = find_header_value(&headers, &["tenant"]).unwrap_or_else(|| "JD".to_string());

    Ok(JoyCodeAuth {
        pt_key,
        login_type,
        tenant,
    })
}

fn collect_joycode_headers_from_json_str(raw: Option<&str>, headers: &mut Vec<(String, String)>) {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return;
    };
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return;
    };
    collect_joycode_headers_from_json_value(&value, headers);
}

fn collect_joycode_headers_from_json_value(value: &Value, headers: &mut Vec<(String, String)>) {
    for key in [
        "cookie",
        "joycode_cookie",
        "joycodeCookie",
        "pt_key",
        "ptKey",
        "ptkey",
        "x-pt-key",
        "loginType",
        "tenant",
    ] {
        if let Some(value) = value.get(key).and_then(Value::as_str) {
            headers.push((key.to_string(), value.to_string()));
        }
    }

    for section in ["headers", "env", "joycode"] {
        if let Some(object) = value.get(section).and_then(Value::as_object) {
            for (key, value) in object {
                if let Some(value) = value.as_str() {
                    headers.push((key.clone(), value.to_string()));
                }
            }
        }
    }
}

fn collect_joycode_headers_from_env_like(content: &str, headers: &mut Vec<(String, String)>) {
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if !value.is_empty() {
            headers.push((key.trim().to_string(), value));
        }
    }
}

fn collect_joycode_headers_from_toml(content: &str, headers: &mut Vec<(String, String)>) {
    let Ok(doc) = content.parse::<toml::Value>() else {
        return;
    };
    collect_joycode_headers_from_toml_value(&doc, headers);
    for section in ["headers", "env", "joycode"] {
        if let Some(value) = doc.get(section) {
            collect_joycode_headers_from_toml_value(value, headers);
            if section == "joycode" {
                if let Some(value) = value.get("headers") {
                    collect_joycode_headers_from_toml_value(value, headers);
                }
                if let Some(value) = value.get("env") {
                    collect_joycode_headers_from_toml_value(value, headers);
                }
            }
        }
    }
}

fn collect_joycode_headers_from_toml_value(
    value: &toml::Value,
    headers: &mut Vec<(String, String)>,
) {
    let Some(table) = value.as_table() else {
        return;
    };
    for (key, value) in table {
        let value = match value {
            toml::Value::String(value) => Some(value.trim().to_string()),
            toml::Value::Integer(value) => Some(value.to_string()),
            toml::Value::Float(value) => Some(value.to_string()),
            toml::Value::Boolean(value) => Some(value.to_string()),
            _ => None,
        };
        if let Some(value) = value.filter(|value| !value.is_empty()) {
            headers.push((key.clone(), value));
        }
    }
}

fn promote_joycode_cookie_header_pairs(headers: &mut Vec<(String, String)>) {
    let cookie = find_header_value(headers, &["cookie", "joycode_cookie", "joycodeCookie"]);

    if let Some(cookie) = cookie {
        headers.push(("cookie".to_string(), cookie.clone()));
        if let Some(pt_key) = extract_cookie_value(&cookie, "pt_key") {
            headers.push(("pt_key".to_string(), pt_key.clone()));
            headers.push(("x-pt-key".to_string(), pt_key));
        }
    }
}

fn find_header_value(headers: &[(String, String)], keys: &[&str]) -> Option<String> {
    headers
        .iter()
        .rev()
        .find(|(key, _)| {
            keys.iter()
                .any(|candidate| key.eq_ignore_ascii_case(candidate))
        })
        .map(|(_, value)| value.clone())
}

fn extract_cookie_value(cookie: &str, name: &str) -> Option<String> {
    cookie.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        (key == name).then(|| value.to_string())
    })
}

fn normalize_joycode_model_list(value: &Value) -> Result<Vec<FetchedModel>, String> {
    let array = if let Some(array) = value.as_array() {
        array
    } else if let Some(array) = value.get("modelList").and_then(Value::as_array) {
        array
    } else if let Some(array) = value.get("models").and_then(Value::as_array) {
        array
    } else {
        return Ok(Vec::new());
    };

    let mut models: Vec<FetchedModel> = array
        .iter()
        .filter_map(|entry| {
            let label = first_string_field(entry, &["label", "name", "displayName"]);
            let upstream_model =
                first_string_field(entry, &["chatApiModel", "apiModel", "modelName", "model"]);
            let id = label
                .clone()
                .or_else(|| upstream_model.clone())
                .or_else(|| first_string_field(entry, &["id"]))?;
            let hidden = entry
                .get("hidden")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if hidden {
                return None;
            }
            let function_type = first_string_field(entry, &["modelFunctionType"]);
            if !matches!(
                function_type.as_deref(),
                None | Some("ALL") | Some("chat") | Some("agent")
            ) {
                return None;
            }
            let adapter_type = joycode_adapter_type(entry);
            let provider = joycode_provider_from_adapter(adapter_type.as_deref());
            let endpoint = joycode_endpoint_from_adapter(adapter_type.as_deref());
            let auth_header = joycode_auth_header_from_adapter(adapter_type.as_deref());
            let responses_mode = adapter_type
                .as_deref()
                .is_some_and(|adapter| adapter.eq_ignore_ascii_case("openai-response"))
                .then(|| "codex".to_string());
            Some(FetchedModel {
                id,
                owned_by: adapter_type
                    .clone()
                    .or_else(|| first_string_field(entry, &["provider", "owned_by"])),
                display_name: label,
                context_window: integer_field(entry, &["maxTotalTokens", "contextWindow"]),
                upstream_model,
                provider,
                endpoint,
                auth_header,
                responses_mode,
            })
        })
        .collect();

    models.sort_by(|a, b| a.id.cmp(&b.id));
    models.dedup_by(|a, b| a.id == b.id);
    Ok(models)
}

fn joycode_adapter_type(entry: &Value) -> Option<String> {
    first_string_field(entry, &["adapterType", "adapter_type"]).or_else(|| {
        entry
            .get("extJson")
            .and_then(|value| {
                if let Some(raw) = value.as_str() {
                    serde_json::from_str::<Value>(raw).ok()
                } else {
                    Some(value.clone())
                }
            })
            .and_then(|ext| first_string_field(&ext, &["adapterType", "adapter_type"]))
    })
}

fn joycode_provider_from_adapter(adapter_type: Option<&str>) -> Option<String> {
    match adapter_type.map(|value| value.to_ascii_lowercase()).as_deref() {
        Some("anthropic") => Some("anthropic".to_string()),
        Some("openai") | Some("openai-response") => Some("openai".to_string()),
        _ => None,
    }
}

fn joycode_endpoint_from_adapter(adapter_type: Option<&str>) -> Option<String> {
    match adapter_type.map(|value| value.to_ascii_lowercase()).as_deref() {
        Some("anthropic") => Some("/v1/messages".to_string()),
        Some("openai-response") => Some("/v1/responses".to_string()),
        Some("openai") => Some("/v1/chat/completions".to_string()),
        _ => None,
    }
}

fn joycode_auth_header_from_adapter(adapter_type: Option<&str>) -> Option<String> {
    match adapter_type.map(|value| value.to_ascii_lowercase()).as_deref() {
        Some("anthropic") => Some("x-api-key".to_string()),
        Some("openai") | Some("openai-response") => Some("bearer".to_string()),
        _ => None,
    }
}

fn integer_field(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|value| {
            value.as_u64().or_else(|| {
                value
                    .as_str()
                    .and_then(|raw| raw.trim().parse::<u64>().ok())
            })
        })
    })
}

fn first_string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    })
}

fn read_vscode_joycode_global_state() -> Result<Value, String> {
    let mut errors = Vec::new();
    for db_path in vscode_global_state_db_candidates() {
        if !db_path.exists() {
            continue;
        }
        match read_joycode_state_from_vscode_db(&db_path) {
            Ok(value) => return Ok(value),
            Err(err) => errors.push(format!("{}: {err}", db_path.display())),
        }
    }

    Err(if errors.is_empty() {
        "未找到 VS Code globalStorage/state.vscdb；请确认已安装并登录官方 JoyCode VS Code 插件"
            .to_string()
    } else {
        format!("读取 VS Code JoyCode 登录态失败：{}", errors.join("; "))
    })
}

fn vscode_global_state_db_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(home) = dirs::home_dir() {
        #[cfg(target_os = "macos")]
        {
            candidates
                .push(home.join("Library/Application Support/Code/User/globalStorage/state.vscdb"));
            candidates.push(home.join(
                "Library/Application Support/Code - Insiders/User/globalStorage/state.vscdb",
            ));
        }
        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = std::env::var("APPDATA") {
                let appdata = PathBuf::from(appdata);
                candidates.push(appdata.join("Code/User/globalStorage/state.vscdb"));
                candidates.push(appdata.join("Code - Insiders/User/globalStorage/state.vscdb"));
            }
        }
        #[cfg(target_os = "linux")]
        {
            candidates.push(home.join(".config/Code/User/globalStorage/state.vscdb"));
            candidates.push(home.join(".config/Code - Insiders/User/globalStorage/state.vscdb"));
        }
    }
    candidates
}

fn read_joycode_state_from_vscode_db(db_path: &PathBuf) -> Result<Value, String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("打开 SQLite 失败: {e}"))?;

    let value: String = conn
        .query_row(
            "SELECT value FROM ItemTable WHERE key = 'JoyCoder.joycoder-fe'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("读取 JoyCoder.joycoder-fe 失败: {e}"))?;

    serde_json::from_str(&value).map_err(|e| format!("解析 JoyCode 插件状态失败: {e}"))
}

fn parse_joycode_login_state(value: &Value) -> Result<JoyCodeLoginState, String> {
    let pt_key = first_string_field(value, &["ptKey", "pt_key", "ptkey"])
        .or_else(|| {
            first_string_field(value, &["cookiesStr"])
                .and_then(|cookie| extract_cookie_value(&cookie, "pt_key"))
        })
        .ok_or_else(|| "VS Code JoyCode 登录态缺少 ptKey，请重新登录 JoyCode".to_string())?;
    let login_type =
        first_string_field(value, &["loginType", "logintype"]).unwrap_or_else(|| "ERP".to_string());
    let tenant = first_string_field(value, &["tenant"]).unwrap_or_else(|| "JD".to_string());
    Ok(JoyCodeLoginState {
        user_name: first_string_field(value, &["userName", "erp", "realName"]),
        tenant,
        login_type,
        pt_key,
        master_base_url: first_string_field(value, &["masterBaseUrl", "master_base_url"]),
        org_full_name: first_string_field(value, &["orgFullName", "org_full_name"]),
        user_id: first_string_field(value, &["userId", "user_id", "erp"]),
    })
}

/// 若 baseURL 以任一已知兼容子路径结尾，返回剥离后的剩余部分；否则 `None`。
///
/// 依赖 [`KNOWN_COMPAT_SUFFIXES`] 按长度降序排列，确保最长前缀优先命中
/// （否则 `/anthropic` 会提前匹配掉 `/api/anthropic` 的场景）。
fn strip_compat_suffix(base_url: &str) -> Option<&str> {
    for suffix in KNOWN_COMPAT_SUFFIXES {
        if base_url.ends_with(*suffix) {
            return Some(&base_url[..base_url.len() - suffix.len()]);
        }
    }
    None
}

/// 判断 baseURL 是否以 OpenAI 风格的版本段 `/v{N}` 结尾（`N` 为一个或多个数字），
/// 例如 `/v1`、`.../paas/v4`。这类 URL 版本号已在路径中，模型端点应为
/// `{base}/models`，不能再补 `/v1`（智谱 Coding Plan 即 `.../coding/paas/v4`）。
fn ends_with_version_segment(url: &str) -> bool {
    let last = url.rsplit('/').next().unwrap_or("");
    last.strip_prefix('v')
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidates_plain_root() {
        let c = build_models_url_candidates("https://api.siliconflow.cn", false, None).unwrap();
        assert_eq!(c, vec!["https://api.siliconflow.cn/v1/models"]);
    }

    #[test]
    fn test_candidates_trailing_slash() {
        let c = build_models_url_candidates("https://api.example.com/", false, None).unwrap();
        assert_eq!(c, vec!["https://api.example.com/v1/models"]);
    }

    #[test]
    fn test_candidates_with_v1() {
        let c = build_models_url_candidates("https://api.example.com/v1", false, None).unwrap();
        assert_eq!(c, vec!["https://api.example.com/v1/models"]);
    }

    #[test]
    fn test_candidates_zhipu_coding_paas_v4() {
        // 智谱 Coding Plan 端点以 /v4 版本段结尾：模型端点是 {base}/models，
        // 正确路径必须排在 .../v4/v1/models（404）之前。
        let c =
            build_models_url_candidates("https://open.bigmodel.cn/api/coding/paas/v4", false, None)
                .unwrap();
        assert_eq!(
            c,
            vec![
                "https://open.bigmodel.cn/api/coding/paas/v4/models",
                "https://open.bigmodel.cn/api/coding/paas/v4/v1/models",
            ]
        );
    }

    #[test]
    fn test_candidates_zai_coding_paas_v4() {
        let c = build_models_url_candidates("https://api.z.ai/api/coding/paas/v4", false, None)
            .unwrap();
        assert_eq!(
            c,
            vec![
                "https://api.z.ai/api/coding/paas/v4/models",
                "https://api.z.ai/api/coding/paas/v4/v1/models",
            ]
        );
    }

    #[test]
    fn test_ends_with_version_segment() {
        assert!(ends_with_version_segment("https://x.com/v1"));
        assert!(ends_with_version_segment(
            "https://open.bigmodel.cn/api/coding/paas/v4"
        ));
        assert!(ends_with_version_segment("https://x.com/v10"));
        assert!(!ends_with_version_segment("https://x.com/api"));
        assert!(!ends_with_version_segment("https://x.com/vX"));
        assert!(!ends_with_version_segment("https://x.com/models"));
        assert!(!ends_with_version_segment("https://api.siliconflow.cn"));
    }

    #[test]
    fn test_candidates_full_url() {
        let c = build_models_url_candidates(
            "https://proxy.example.com/v1/chat/completions",
            true,
            None,
        )
        .unwrap();
        assert_eq!(c, vec!["https://proxy.example.com/v1/models"]);
    }

    #[test]
    fn test_candidates_empty() {
        assert!(build_models_url_candidates("", false, None).is_err());
    }

    #[test]
    fn test_candidates_override_returns_single() {
        let c = build_models_url_candidates(
            "https://api.deepseek.com/anthropic",
            false,
            Some("https://api.deepseek.com/models"),
        )
        .unwrap();
        assert_eq!(c, vec!["https://api.deepseek.com/models"]);
    }

    #[test]
    fn test_candidates_override_empty_falls_through() {
        let c =
            build_models_url_candidates("https://api.siliconflow.cn", false, Some("   ")).unwrap();
        assert_eq!(c, vec!["https://api.siliconflow.cn/v1/models"]);
    }

    #[test]
    fn test_candidates_deepseek_strip_anthropic() {
        let c =
            build_models_url_candidates("https://api.deepseek.com/anthropic", false, None).unwrap();
        assert_eq!(
            c,
            vec![
                "https://api.deepseek.com/anthropic/v1/models",
                "https://api.deepseek.com/v1/models",
                "https://api.deepseek.com/models",
            ]
        );
    }

    #[test]
    fn test_candidates_zhipu_strip_api_anthropic() {
        let c = build_models_url_candidates("https://open.bigmodel.cn/api/anthropic", false, None)
            .unwrap();
        assert_eq!(
            c,
            vec![
                "https://open.bigmodel.cn/api/anthropic/v1/models",
                "https://open.bigmodel.cn/v1/models",
                "https://open.bigmodel.cn/models",
            ]
        );
    }

    #[test]
    fn test_candidates_bailian_strip_apps_anthropic() {
        let c = build_models_url_candidates(
            "https://dashscope.aliyuncs.com/apps/anthropic",
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            c,
            vec![
                "https://dashscope.aliyuncs.com/apps/anthropic/v1/models",
                "https://dashscope.aliyuncs.com/v1/models",
                "https://dashscope.aliyuncs.com/models",
            ]
        );
    }

    #[test]
    fn test_candidates_stepfun_strip_step_plan() {
        let c =
            build_models_url_candidates("https://api.stepfun.com/step_plan", false, None).unwrap();
        assert_eq!(
            c,
            vec![
                "https://api.stepfun.com/step_plan/v1/models",
                "https://api.stepfun.com/v1/models",
                "https://api.stepfun.com/models",
            ]
        );
    }

    #[test]
    fn test_candidates_doubao_strip_api_coding() {
        let c = build_models_url_candidates(
            "https://ark.cn-beijing.volces.com/api/coding",
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            c,
            vec![
                "https://ark.cn-beijing.volces.com/api/coding/v1/models",
                "https://ark.cn-beijing.volces.com/v1/models",
                "https://ark.cn-beijing.volces.com/models",
            ]
        );
    }

    #[test]
    fn test_candidates_rightcode_strip_claude() {
        let c = build_models_url_candidates("https://www.right.codes/claude", false, None).unwrap();
        assert_eq!(
            c,
            vec![
                "https://www.right.codes/claude/v1/models",
                "https://www.right.codes/v1/models",
                "https://www.right.codes/models",
            ]
        );
    }

    #[test]
    fn test_joycode_color_gateway_sign_matches_official_algorithm() {
        let sign = joycode_color_gateway_sign(&[
            ("functionId", "joycode_modelList".to_string()),
            ("appid", "joycode_ide".to_string()),
            ("t", "1700000000000".to_string()),
        ]);
        assert_eq!(
            sign,
            "75c751241df294d1fc67f241bdbd535e07b95a7b5c95a99a7b78464d5d9b93a7"
        );
    }

    #[test]
    fn test_collect_joycode_auth_from_headers_json_cookie() {
        let auth = collect_joycode_auth(Some(
            r#"{"cookie":"pt_key=abc; other=1","loginType":"N_PIN_PC","tenant":"JD"}"#,
        ))
        .unwrap();
        assert_eq!(
            auth,
            JoyCodeAuth {
                pt_key: "abc".to_string(),
                login_type: "N_PIN_PC".to_string(),
                tenant: "JD".to_string(),
            }
        );
    }

    #[test]
    fn test_normalize_joycode_model_list_filters_and_sorts() {
        let models = normalize_joycode_model_list(&json!([
            {
                "label": "Claude-Sonnet-4.6-hq",
                "chatApiModel": "claude-sonnet-4-6",
                "modelFunctionType": "agent",
                "maxTotalTokens": 256000,
                "extJson": "{\"adapterType\":\"anthropic\"}"
            },
            {
                "label": "Hidden",
                "hidden": true
            },
            {
                "label": "Embedding",
                "modelFunctionType": "embedding"
            },
            {
                "model": "GPT 5.3-codex"
            }
        ]))
        .unwrap();
        assert_eq!(
            models,
            vec![
                FetchedModel {
                    id: "Claude-Sonnet-4.6-hq".to_string(),
                    owned_by: Some("anthropic".to_string()),
                    display_name: Some("Claude-Sonnet-4.6-hq".to_string()),
                    context_window: Some(256000),
                    upstream_model: Some("claude-sonnet-4-6".to_string()),
                    provider: Some("anthropic".to_string()),
                    endpoint: Some("/v1/messages".to_string()),
                    auth_header: Some("x-api-key".to_string()),
                    responses_mode: None,
                },
                FetchedModel {
                    id: "GPT 5.3-codex".to_string(),
                    owned_by: None,
                    display_name: None,
                    context_window: None,
                    upstream_model: Some("GPT 5.3-codex".to_string()),
                    provider: None,
                    endpoint: None,
                    auth_header: None,
                    responses_mode: None,
                },
            ]
        );
    }

    #[test]
    fn test_parse_joycode_login_state_from_vscode_state() {
        let state = parse_joycode_login_state(&json!({
            "userName": "hechengjun.9",
            "ptKey": "BJ.token",
            "loginType": "ERP",
            "tenant": "JD",
            "masterBaseUrl": "http://joycode-api-saas.jd.com",
            "orgFullName": "京东集团",
            "userId": "hechengjun.9"
        }))
        .unwrap();
        assert_eq!(state.user_name.as_deref(), Some("hechengjun.9"));
        assert_eq!(state.pt_key, "BJ.token");
        assert_eq!(state.login_type, "ERP");
        assert_eq!(state.tenant, "JD");
        assert_eq!(
            state.master_base_url.as_deref(),
            Some("http://joycode-api-saas.jd.com")
        );
        assert_eq!(state.org_full_name.as_deref(), Some("京东集团"));
        assert_eq!(state.user_id.as_deref(), Some("hechengjun.9"));
    }

    #[test]
    fn test_parse_joycode_login_state_falls_back_to_cookie_pt_key() {
        let state = parse_joycode_login_state(&json!({
            "cookiesStr": "a=1; pt_key=BJ.from-cookie; b=2"
        }))
        .unwrap();
        assert_eq!(state.pt_key, "BJ.from-cookie");
        assert_eq!(state.login_type, "ERP");
        assert_eq!(state.tenant, "JD");
    }

    #[test]
    fn test_candidates_longer_suffix_wins() {
        // baseURL 以 /api/anthropic 结尾时，应剥离整个 /api/anthropic，
        // 而不是只剥离 /anthropic（那样会得到残缺的 https://.../api 根）。
        let c = build_models_url_candidates("https://api.z.ai/api/anthropic", false, None).unwrap();
        assert_eq!(
            c,
            vec![
                "https://api.z.ai/api/anthropic/v1/models",
                "https://api.z.ai/v1/models",
                "https://api.z.ai/models",
            ]
        );
    }

    #[test]
    fn test_candidates_no_suffix_no_strip() {
        let c = build_models_url_candidates("https://openrouter.ai/api", false, None).unwrap();
        assert_eq!(c, vec!["https://openrouter.ai/api/v1/models"]);
    }

    #[test]
    fn test_candidates_deduplicate() {
        // 虚构 case：baseURL 就是 "scheme://host"，剥不出子路径，应只有一个候选。
        let c = build_models_url_candidates("https://host.example.com", false, None).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_parse_response() {
        let json = r#"{"object":"list","data":[{"id":"gpt-4","object":"model","owned_by":"openai"},{"id":"claude-3-sonnet","object":"model","owned_by":"anthropic"}]}"#;
        let resp: ModelsResponse = serde_json::from_str(json).unwrap();
        let data = resp.data.unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].id, "gpt-4");
        assert_eq!(data[0].owned_by.as_deref(), Some("openai"));
        assert_eq!(data[1].id, "claude-3-sonnet");
    }

    #[test]
    fn test_parse_response_no_owned_by() {
        let json = r#"{"object":"list","data":[{"id":"my-model","object":"model"}]}"#;
        let resp: ModelsResponse = serde_json::from_str(json).unwrap();
        let data = resp.data.unwrap();
        assert_eq!(data[0].id, "my-model");
        assert!(data[0].owned_by.is_none());
    }

    #[test]
    fn test_parse_response_empty_data() {
        let json = r#"{"object":"list","data":[]}"#;
        let resp: ModelsResponse = serde_json::from_str(json).unwrap();
        assert!(resp.data.unwrap().is_empty());
    }
}

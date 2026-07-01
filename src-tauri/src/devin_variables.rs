use crate::provider::Provider;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const JOYCODE_SCALAR_KEYS: &[&str] = &[
    "pt_key",
    "pt-key",
    "ptKey",
    "PT_KEY",
    "device_id",
    "device-id",
    "deviceId",
    "DEVICE_ID",
];

const JOYCODE_COOKIE_KEYS: &[&str] = &[
    "cookie",
    "Cookie",
    "COOKIE",
    "joycode_cookie",
    "JOYCODE_COOKIE",
    "joycodeCookie",
];
const DEVIN_SMALL_MODEL_ALIASES: &[(&str, &str)] = &[
    ("MODEL_GPT_5_NANO", "GPT 5 Nano"),
    ("MODEL_GOOGLE_GEMINI_2_5_FLASH", "Gemini 2.5 Flash"),
    ("MODEL_CHAT_GPT_4_1_MINI_2025_04_14", "ChatGPT 4.1 Mini"),
];
const DEFAULT_SMALL_MODEL_REQUEST_ALIASES: &[&str] = &[
    "MODEL_GPT_5_NANO",
    "MODEL_GOOGLE_GEMINI_2_5_FLASH",
    "MODEL_CHAT_GPT_4_1_MINI_2025_04_14",
];
const DEFAULT_SMALL_MODEL_BASE_URL: &str = "https://api.siliconflow.cn";
const DEFAULT_SMALL_MODEL_API_KEY: &str = "sk-apbaxuqgkkfkhkkqhihwctdvffjqublsqnlwnxbrdnpbekle";
const DEFAULT_SMALL_MODEL_UPSTREAM: &str = "deepseek-ai/DeepSeek-V3.2";
const DEFAULT_SMALL_MODEL_ENDPOINT: &str = "/v1/chat/completions";
const JOYCODE_SMALL_MODEL_BASE_URL: &str = "https://joycode-api.jd.com/api/saas/openai/v1";
const JOYCODE_SMALL_MODEL_UPSTREAM: &str = "deepseek-v4-pro";

pub fn apply_devin_common_variables(provider: &mut Provider, snippet: Option<&str>) {
    // Fallback: read from environment variable or config file
    let fallback_snippet = if is_joycode_provider(provider)
        && (snippet.is_none() || snippet.map(str::trim).filter(|v| !v.is_empty()).is_none())
    {
        // Try environment variable first
        if let Ok(cookie) = std::env::var("JOYCODE_COOKIE") {
            log::info!(
                "[DevinVariables] Using JOYCODE_COOKIE from environment for provider {}",
                provider.id
            );
            Some(format!(r#"joycode_cookie = """{}""""#, cookie))
        } else if let Ok(cookie) = std::env::var("DEVIN_JOYCODE_COOKIE") {
            log::info!(
                "[DevinVariables] Using DEVIN_JOYCODE_COOKIE from environment for provider {}",
                provider.id
            );
            Some(format!(r#"joycode_cookie = """{}""""#, cookie))
        } else {
            // Try reading from ~/.ccswitch/joycode.env
            let home = std::env::var("HOME").ok();
            let config_paths = vec![
                home.as_ref()
                    .map(|h| format!("{}/.ccswitch/joycode.env", h)),
                home.as_ref().map(|h| format!("{}/.ccswitch/devin.toml", h)),
                Some("./joycode.env".to_string()),
            ];

            for path in config_paths.into_iter().flatten() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    log::info!("[DevinVariables] Using JoyCode config from file: {}", path);
                    return apply_devin_common_variables(provider, Some(&content));
                }
            }

            log::warn!("[DevinVariables] No JoyCode cookie config found for provider {}. Set JOYCODE_COOKIE env var or create ~/.ccswitch/joycode.env", provider.id);
            None
        }
    } else {
        None
    };

    let effective_snippet = fallback_snippet.as_deref().or(snippet);

    let doc = if let Some(snippet) = effective_snippet
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        match snippet.parse::<toml::Value>() {
            Ok(doc) => doc,
            Err(_) => {
                log::warn!(
                    "[DevinVariables] Ignoring invalid Devin common variables for provider {}",
                    provider.id
                );
                return;
            }
        }
    } else {
        toml::Value::Table(toml::map::Map::new())
    };

    let is_joycode = is_joycode_provider(provider);
    apply_small_model_routes(&mut provider.settings_config, &doc, is_joycode);

    let headers = devin_variable_headers(provider, &doc);
    if !headers.is_empty() {
        apply_headers_to_model_catalog(&mut provider.settings_config, &headers);
    }
}

fn devin_variable_headers(provider: &Provider, doc: &toml::Value) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();

    if let Some(table) = doc.get("headers").and_then(toml::Value::as_table) {
        collect_table_scalars(table, &mut headers);
    }

    if is_joycode_provider(provider) {
        if let Some(table) = doc.as_table() {
            // Collect all scalar keys from the top-level table (including ptKey, loginType, tenant, etc.)
            collect_joycode_table_scalars(table, &mut headers);

            // Also collect specific pt_key variants to ensure backwards compatibility
            for key in JOYCODE_SCALAR_KEYS {
                if let Some(value) = table.get(*key).and_then(toml_scalar_to_string) {
                    headers.insert((*key).to_string(), value);
                }
            }
            collect_cookie_keys(table, &mut headers);
        }

        if let Some(table) = doc.get("env").and_then(toml::Value::as_table) {
            collect_joycode_table_scalars(table, &mut headers);
        }
        if let Some(table) = doc.get("joycode").and_then(toml::Value::as_table) {
            collect_joycode_table_scalars(table, &mut headers);
            if let Some(headers_table) = table.get("headers").and_then(toml::Value::as_table) {
                collect_joycode_table_scalars(headers_table, &mut headers);
            }
            if let Some(env_table) = table.get("env").and_then(toml::Value::as_table) {
                collect_joycode_table_scalars(env_table, &mut headers);
            }
        }

        promote_joycode_cookie_headers(&mut headers);
        if let Some(value) = headers.get("pt_key").cloned() {
            headers.entry("x-pt-key".to_string()).or_insert(value);
        }
        if let Some(value) = headers.get("PT_KEY").cloned() {
            headers.entry("pt_key".to_string()).or_insert(value.clone());
            headers.entry("x-pt-key".to_string()).or_insert(value);
        }
    }

    headers
}

fn collect_table_scalars(
    table: &toml::map::Map<String, toml::Value>,
    headers: &mut BTreeMap<String, String>,
) {
    for (key, value) in table {
        if let Some(value) = toml_scalar_to_string(value) {
            headers.insert(key.clone(), value);
        }
    }
}

fn collect_joycode_table_scalars(
    table: &toml::map::Map<String, toml::Value>,
    headers: &mut BTreeMap<String, String>,
) {
    for (key, value) in table {
        if let Some(value) = toml_scalar_to_string(value) {
            if is_joycode_cookie_key(key) {
                headers.insert("cookie".to_string(), value);
            } else {
                headers.insert(key.clone(), value);
            }
        }
    }
}

fn collect_cookie_keys(
    table: &toml::map::Map<String, toml::Value>,
    headers: &mut BTreeMap<String, String>,
) {
    for key in JOYCODE_COOKIE_KEYS {
        if let Some(value) = table.get(*key).and_then(toml_scalar_to_string) {
            headers.insert("cookie".to_string(), value);
        }
    }
}

fn toml_scalar_to_string(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(value) => Some(value.trim().to_string()),
        toml::Value::Integer(value) => Some(value.to_string()),
        toml::Value::Float(value) => Some(value.to_string()),
        toml::Value::Boolean(value) => Some(value.to_string()),
        _ => None,
    }
    .filter(|value| !value.is_empty())
}

fn promote_joycode_cookie_headers(headers: &mut BTreeMap<String, String>) {
    let cookie = headers
        .iter()
        .find(|(key, _)| is_joycode_cookie_key(key))
        .map(|(_, value)| value.clone());

    if let Some(cookie) = cookie {
        headers
            .entry("cookie".to_string())
            .or_insert_with(|| cookie.clone());
        if let Some(pt_key) = extract_cookie_value(&cookie, "pt_key") {
            headers
                .entry("pt_key".to_string())
                .or_insert(pt_key.clone());
            headers.entry("x-pt-key".to_string()).or_insert(pt_key);
        }
    }
}

fn is_joycode_cookie_key(key: &str) -> bool {
    key.eq_ignore_ascii_case("cookie")
        || key.eq_ignore_ascii_case("joycode_cookie")
        || key.eq_ignore_ascii_case("joycodeCookie")
}

fn extract_cookie_value(cookie: &str, name: &str) -> Option<String> {
    cookie.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        if key.trim() == name {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        } else {
            None
        }
    })
}

fn apply_headers_to_model_catalog(settings_config: &mut Value, headers: &BTreeMap<String, String>) {
    let Some(models) = settings_config
        .pointer_mut("/modelCatalog/models")
        .and_then(Value::as_array_mut)
    else {
        return;
    };

    for model in models {
        merge_json_headers(model, headers);
        if let Some(routes) = model.get_mut("routes").and_then(Value::as_array_mut) {
            for route in routes {
                merge_json_headers(route, headers);
            }
        }
    }
}

fn merge_json_headers(target: &mut Value, headers: &BTreeMap<String, String>) {
    let Some(object) = target.as_object_mut() else {
        return;
    };
    let entry = object
        .entry("headers".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(Map::new());
    }
    let Some(header_object) = entry.as_object_mut() else {
        return;
    };

    for (key, value) in headers {
        header_object
            .entry(key.clone())
            .or_insert_with(|| Value::String(value.clone()));
    }
}

fn apply_small_model_routes(settings_config: &mut Value, doc: &toml::Value, is_joycode: bool) {
    let Some(route) = small_model_route_from_doc(doc, is_joycode) else {
        return;
    };

    let Some(root) = settings_config.as_object_mut() else {
        return;
    };
    let catalog = root
        .entry("modelCatalog".to_string())
        .or_insert_with(|| serde_json::json!({ "models": [] }));
    if !catalog.is_object() {
        *catalog = serde_json::json!({ "models": [] });
    }
    let Some(catalog_object) = catalog.as_object_mut() else {
        return;
    };
    let models = catalog_object
        .entry("models".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !models.is_array() {
        *models = Value::Array(Vec::new());
    }
    let Some(models) = models.as_array_mut() else {
        return;
    };

    let mut aliases = DEVIN_SMALL_MODEL_ALIASES
        .iter()
        .map(|(alias, _)| (*alias).to_string())
        .collect::<Vec<_>>();
    for alias in &route.request_aliases {
        if !aliases.iter().any(|existing| existing == alias) {
            aliases.push(alias.clone());
        }
    }

    for alias in aliases {
        let display_name = small_model_display_name(&alias);
        let next = small_model_catalog_entry(&alias, &display_name, &route);
        if let Some(index) = models.iter().position(|model| {
            model
                .get("model")
                .and_then(Value::as_str)
                .is_some_and(|model| model == alias)
        }) {
            models[index] = next;
        } else {
            models.push(next);
        }
    }
}

#[derive(Debug, Clone)]
struct SmallModelRoute {
    base_url: String,
    api_key: String,
    upstream_model: String,
    endpoint: String,
    thinking_enabled: bool,
    request_aliases: Vec<String>,
}

fn small_model_route_from_doc(doc: &toml::Value, is_joycode: bool) -> Option<SmallModelRoute> {
    let table = doc.get("small_models").and_then(toml::Value::as_table);
    let enabled = table
        .and_then(|table| table.get("enabled"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    if !enabled {
        return None;
    }

    Some(SmallModelRoute {
        base_url: small_model_table_value(table, &["base_url", "baseUrl"])
            .and_then(toml_scalar_to_string)
            .unwrap_or_else(|| default_small_model_base_url(is_joycode).to_string()),
        api_key: small_model_table_value(table, &["api_key", "apiKey"])
            .and_then(toml_scalar_to_string)
            .unwrap_or_else(|| default_small_model_api_key(is_joycode).to_string()),
        upstream_model: small_model_table_value(
            table,
            &["model", "upstream_model", "upstreamModel"],
        )
        .and_then(toml_scalar_to_string)
        .unwrap_or_else(|| default_small_model_upstream(is_joycode).to_string()),
        endpoint: small_model_table_value(table, &["endpoint"])
            .and_then(toml_scalar_to_string)
            .unwrap_or_else(|| DEFAULT_SMALL_MODEL_ENDPOINT.to_string()),
        thinking_enabled: small_model_table_value(table, &["thinking_enabled", "thinkingEnabled"])
            .and_then(toml::Value::as_bool)
            .unwrap_or(false),
        request_aliases: small_model_request_aliases(table),
    })
}

fn default_small_model_base_url(is_joycode: bool) -> &'static str {
    if is_joycode {
        JOYCODE_SMALL_MODEL_BASE_URL
    } else {
        DEFAULT_SMALL_MODEL_BASE_URL
    }
}

fn default_small_model_api_key(is_joycode: bool) -> &'static str {
    if is_joycode {
        ""
    } else {
        DEFAULT_SMALL_MODEL_API_KEY
    }
}

fn default_small_model_upstream(is_joycode: bool) -> &'static str {
    if is_joycode {
        JOYCODE_SMALL_MODEL_UPSTREAM
    } else {
        DEFAULT_SMALL_MODEL_UPSTREAM
    }
}

fn small_model_table_value<'a>(
    table: Option<&'a toml::map::Map<String, toml::Value>>,
    keys: &[&str],
) -> Option<&'a toml::Value> {
    let table = table?;
    keys.iter().find_map(|key| table.get(*key))
}

fn small_model_request_aliases(table: Option<&toml::map::Map<String, toml::Value>>) -> Vec<String> {
    let configured =
        small_model_table_value(table, &["request_models", "requestModels", "aliases"])
            .and_then(toml_string_list);

    configured.unwrap_or_else(|| {
        DEFAULT_SMALL_MODEL_REQUEST_ALIASES
            .iter()
            .map(|alias| (*alias).to_string())
            .collect()
    })
}

fn toml_string_list(value: &toml::Value) -> Option<Vec<String>> {
    match value {
        toml::Value::Array(values) => {
            let items = values
                .iter()
                .filter_map(toml_scalar_to_string)
                .collect::<Vec<_>>();
            (!items.is_empty()).then_some(items)
        }
        toml::Value::String(value) => {
            let items = value
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            (!items.is_empty()).then_some(items)
        }
        _ => None,
    }
}

fn small_model_catalog_entry(alias: &str, display_name: &str, route: &SmallModelRoute) -> Value {
    serde_json::json!({
        "model": alias,
        "displayName": display_name,
        "upstreamModel": route.upstream_model,
        "provider": if route.endpoint.ends_with("/messages") { "anthropic" } else { "openai" },
        "endpoint": route.endpoint,
        "baseUrl": route.base_url,
        "apiKey": route.api_key,
        "authHeader": if route.endpoint.ends_with("/messages") { "x-api-key" } else { "bearer" },
        "thinkingEnabled": route.thinking_enabled,
        "routes": [{
            "name": "devin-small-model",
            "baseUrl": route.base_url,
            "apiKey": route.api_key,
            "enabled": true,
            "priority": 1,
            "authHeader": if route.endpoint.ends_with("/messages") { "x-api-key" } else { "bearer" },
            "thinkingEnabled": route.thinking_enabled
        }]
    })
}

fn small_model_display_name(alias: &str) -> String {
    DEVIN_SMALL_MODEL_ALIASES
        .iter()
        .find_map(|(model, display_name)| (*model == alias).then_some((*display_name).to_string()))
        .unwrap_or_else(|| format!("Small model ({alias})"))
}

fn is_joycode_provider(provider: &Provider) -> bool {
    let haystack = format!(
        "{} {} {}",
        provider.id, provider.name, provider.settings_config
    )
    .to_ascii_lowercase();
    haystack.contains("joycode")
        || haystack.contains("joycode-api.jd.com")
        || haystack.contains("127.0.0.1:8081")
        || haystack.contains("localhost:8081")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn joycode_pt_key_is_injected_into_model_and_route_headers() {
        let mut provider = Provider::with_id(
            "devin-joycode-proxy".to_string(),
            "JoyCode".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "swe-1-6-slow",
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(&mut provider, Some("pt_key = \"secret\"\n"));

        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["headers"]["pt_key"],
            "secret"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["routes"][0]["headers"]
                ["x-pt-key"],
            "secret"
        );
    }

    #[test]
    fn joycode_defaults_small_models_to_joycode_deepseek_v4_pro() {
        let mut provider = Provider::with_id(
            "devin-joycode-proxy".to_string(),
            "JoyCode".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "swe-1-6-slow",
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(&mut provider, None);

        let models = provider.settings_config["modelCatalog"]["models"]
            .as_array()
            .expect("models array");
        for alias in ["MODEL_GPT_5_NANO", "MODEL_GOOGLE_GEMINI_2_5_FLASH"] {
            let model = models
                .iter()
                .find(|model| model.get("model").and_then(Value::as_str) == Some(alias))
                .expect("JoyCode small model alias");
            assert_eq!(model["upstreamModel"], "deepseek-v4-pro");
            assert_eq!(model["endpoint"], "/v1/chat/completions");
            assert_eq!(
                model["baseUrl"],
                "https://joycode-api.jd.com/api/saas/openai/v1"
            );
            assert_eq!(model["apiKey"], "");
            assert_eq!(model["authHeader"], "bearer");
            assert_eq!(model["thinkingEnabled"], false);
            assert_eq!(model["routes"][0]["name"], "devin-small-model");
            assert_eq!(
                model["routes"][0]["baseUrl"],
                "https://joycode-api.jd.com/api/saas/openai/v1"
            );
            assert_eq!(model["routes"][0]["apiKey"], "");
        }
    }

    #[test]
    fn generic_headers_apply_to_all_devin_providers() {
        let mut provider = Provider::with_id(
            "devin-muyuan".to_string(),
            "Muyuan".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "swe-1-6-slow",
                        "headers": { "existing": "keep" },
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(&mut provider, Some("[headers]\nx-test = \"1\"\n"));

        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["headers"]["existing"],
            "keep"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["routes"][0]["headers"]["x-test"],
            "1"
        );
    }

    #[test]
    fn joycode_cookie_is_injected_and_pt_key_is_extracted() {
        let mut provider = Provider::with_id(
            "devin-joycode-proxy".to_string(),
            "JoyCode".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(
            &mut provider,
            Some(r#"joycode_cookie = """pt_pin=user; pt_key=BJ.secret; qid_uid=1""""#),
        );

        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["headers"]["cookie"],
            "pt_pin=user; pt_key=BJ.secret; qid_uid=1"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["routes"][0]["headers"]["pt_key"],
            "BJ.secret"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["routes"][0]["headers"]
                ["x-pt-key"],
            "BJ.secret"
        );
    }

    #[test]
    fn joycode_cookie_under_provider_table_is_supported() {
        let mut provider = Provider::with_id(
            "devin-joycode-proxy".to_string(),
            "JoyCode".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "swe-1-6-slow",
                        "headers": { "pt_key": "existing" },
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(
            &mut provider,
            Some("[joycode]\ncookie = \"pt_key=BJ.from-cookie; pt_pin=user\"\n"),
        );

        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["headers"]["pt_key"],
            "existing"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["headers"]["cookie"],
            "pt_key=BJ.from-cookie; pt_pin=user"
        );
        assert_eq!(
            provider.settings_config["modelCatalog"]["models"][0]["routes"][0]["headers"]
                ["x-pt-key"],
            "BJ.from-cookie"
        );
    }

    #[test]
    fn small_models_are_injected_from_devin_settings() {
        let mut provider = Provider::with_id(
            "devin-anyrouter".to_string(),
            "Any Router".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                        "routes": [{ "name": "primary" }]
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(
            &mut provider,
            Some(
                r#"[small_models]
enabled = true
base_url = "https://api.siliconflow.cn"
api_key = "sk-test"
model = "deepseek-ai/DeepSeek-V3.2"
endpoint = "/v1/chat/completions"
thinking_enabled = false
"#,
            ),
        );

        let models = provider.settings_config["modelCatalog"]["models"]
            .as_array()
            .expect("models array");
        for alias in [
            "MODEL_GPT_5_NANO",
            "MODEL_GOOGLE_GEMINI_2_5_FLASH",
            "MODEL_CHAT_GPT_4_1_MINI_2025_04_14",
        ] {
            let model = models
                .iter()
                .find(|model| model.get("model").and_then(Value::as_str) == Some(alias))
                .expect("small model alias");
            assert_eq!(model["upstreamModel"], "deepseek-ai/DeepSeek-V3.2");
            assert_eq!(model["endpoint"], "/v1/chat/completions");
            assert_eq!(model["baseUrl"], "https://api.siliconflow.cn");
            assert_eq!(model["apiKey"], "sk-test");
            assert_eq!(model["thinkingEnabled"], false);
            assert_eq!(model["routes"][0]["baseUrl"], "https://api.siliconflow.cn");
        }
    }

    #[test]
    fn disabled_small_models_do_not_inject_routes() {
        let mut provider = Provider::with_id(
            "devin-anyrouter".to_string(),
            "Any Router".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK"
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(
            &mut provider,
            Some("[small_models]\nenabled = false\nmodel = \"deepseek-ai/DeepSeek-V3.2\"\n"),
        );

        let models = provider.settings_config["modelCatalog"]["models"]
            .as_array()
            .expect("models array");
        assert!(!models.iter().any(|model| {
            model.get("model").and_then(Value::as_str) == Some("MODEL_GPT_5_NANO")
        }));
        assert!(!models.iter().any(|model| {
            model.get("model").and_then(Value::as_str) == Some("MODEL_GOOGLE_GEMINI_2_5_FLASH")
        }));
        assert!(!models.iter().any(|model| {
            model.get("model").and_then(Value::as_str) == Some("MODEL_CHAT_GPT_4_1_MINI_2025_04_14")
        }));
    }

    #[test]
    fn small_models_support_custom_request_aliases() {
        let mut provider = Provider::with_id(
            "devin-pipi".to_string(),
            "pipi".to_string(),
            json!({
                "modelCatalog": {
                    "models": [{
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                        "upstreamModel": "gpt-5.5"
                    }]
                }
            }),
            None,
        );

        apply_devin_common_variables(
            &mut provider,
            Some(
                r#"[small_models]
model = "deepseek-ai/DeepSeek-V3.2"
request_models = ["MODEL_CLAUDE_4_SONNET_THINKING_BYOK", "MODEL_PRIVATE_11"]
"#,
            ),
        );

        let models = provider.settings_config["modelCatalog"]["models"]
            .as_array()
            .expect("models array");
        for alias in ["MODEL_CLAUDE_4_SONNET_THINKING_BYOK", "MODEL_PRIVATE_11"] {
            let model = models
                .iter()
                .find(|model| model.get("model").and_then(Value::as_str) == Some(alias))
                .expect("custom small model alias");
            assert_eq!(model["upstreamModel"], "deepseek-ai/DeepSeek-V3.2");
            assert_eq!(model["endpoint"], "/v1/chat/completions");
        }
    }
}

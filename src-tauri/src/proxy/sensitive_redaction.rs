use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::OnceLock;

const REDACTED: &str = "<redacted>";
const PSEUDONYM_EMAIL_DOMAIN: &str = "example.invalid";

#[derive(Debug, Clone, Default)]
pub(crate) struct SensitiveRewriteMap {
    replacements: Vec<(String, String)>,
}

impl SensitiveRewriteMap {
    pub(crate) fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }

    fn insert(&mut self, original: &str, pseudonym: String) -> String {
        if let Some((existing, _)) = self
            .replacements
            .iter()
            .find(|(_, mapped_original)| mapped_original == original)
        {
            return existing.clone();
        }
        self.replacements
            .push((pseudonym.clone(), original.to_string()));
        pseudonym
    }

    pub(crate) fn restore_text(&self, input: &str) -> String {
        let mut text = input.to_string();
        for (pseudonym, original) in &self.replacements {
            text = text.replace(pseudonym, original);
        }
        text
    }
}

pub(crate) fn redact_sensitive_text(input: &str) -> String {
    let mut text = input.to_string();
    text = json_string_field_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let value = caps.get(3).map(|m| m.as_str()).unwrap_or_default();
            format!(
                "{}<redacted,len={}>{}",
                &caps[1],
                value.chars().count(),
                &caps[4]
            )
        })
        .to_string();
    text = cookie_pair_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let value = caps.get(3).map(|m| m.as_str()).unwrap_or_default();
            format!("{}{}=<redacted,len={}>", &caps[1], &caps[2], value.len())
        })
        .to_string();
    text = email_regex()
        .replace_all(&text, "<redacted-email>")
        .to_string();
    text = phone_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            format!(
                "{}<redacted-phone>{}",
                caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                caps.get(3).map(|m| m.as_str()).unwrap_or_default()
            )
        })
        .to_string();
    text = cn_id_card_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            format!(
                "{}<redacted-id-card>{}",
                caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                caps.get(3).map(|m| m.as_str()).unwrap_or_default()
            )
        })
        .to_string();
    text = bank_card_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            format!(
                "{}<redacted-bank-card>{}",
                caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                caps.get(3).map(|m| m.as_str()).unwrap_or_default()
            )
        })
        .to_string();
    text
}

pub(crate) fn redact_sensitive_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                if is_sensitive_key(key) {
                    *value = redacted_value(value);
                } else {
                    redact_sensitive_value(value);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_sensitive_value(item);
            }
        }
        Value::String(value) => {
            *value = redact_sensitive_text(value);
        }
        _ => {}
    }
}

pub(crate) fn pseudonymize_sensitive_value(value: &mut Value) -> SensitiveRewriteMap {
    let mut state = SensitivePseudonymizer::default();
    state.pseudonymize_value(value);
    state.into_map()
}

#[derive(Default)]
struct SensitivePseudonymizer {
    map: SensitiveRewriteMap,
    used_pseudonyms: BTreeSet<String>,
}

impl SensitivePseudonymizer {
    fn into_map(self) -> SensitiveRewriteMap {
        self.map
    }

    fn pseudonymize_value(&mut self, value: &mut Value) {
        match value {
            Value::Object(map) => {
                for (key, value) in map.iter_mut() {
                    if is_credential_key(key) && !value.is_array() && !value.is_object() {
                        *value = redacted_value(value);
                    } else {
                        self.pseudonymize_value(value);
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.pseudonymize_value(item);
                }
            }
            Value::String(text) => {
                *text = self.pseudonymize_text(text);
            }
            _ => {}
        }
    }

    fn pseudonymize_text(&mut self, input: &str) -> String {
        let mut text = input.to_string();

        text = cn_id_card_regex()
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                let original = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
                format!(
                    "{}{}{}",
                    caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                    {
                        let pseudonym = self.stable_id_card(original);
                        self.map.insert(original, pseudonym)
                    },
                    caps.get(3).map(|m| m.as_str()).unwrap_or_default()
                )
            })
            .to_string();
        text = bank_card_regex()
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                let original = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
                format!(
                    "{}{}{}",
                    caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                    {
                        let pseudonym = self.stable_bank_card(original);
                        self.map.insert(original, pseudonym)
                    },
                    caps.get(3).map(|m| m.as_str()).unwrap_or_default()
                )
            })
            .to_string();
        text = email_regex()
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                let original = caps.get(0).map(|m| m.as_str()).unwrap_or_default();
                let pseudonym = self.stable_email(original);
                self.map.insert(original, pseudonym)
            })
            .to_string();
        text = phone_regex()
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                let original = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
                format!(
                    "{}{}{}",
                    caps.get(1).map(|m| m.as_str()).unwrap_or_default(),
                    {
                        let pseudonym = self.stable_phone(original);
                        self.map.insert(original, pseudonym)
                    },
                    caps.get(3).map(|m| m.as_str()).unwrap_or_default()
                )
            })
            .to_string();

        text
    }

    fn stable_email(&mut self, original: &str) -> String {
        let hash = stable_sensitive_hash("email", original);
        self.unique_pseudonym(format!(
            "devin-user-{}@{}",
            &hash[..12],
            PSEUDONYM_EMAIL_DOMAIN
        ))
    }

    fn stable_phone(&mut self, original: &str) -> String {
        let number = stable_sensitive_number("phone", original) % 100_000_000;
        self.unique_pseudonym(format!("139{number:08}"))
    }

    fn stable_id_card(&mut self, original: &str) -> String {
        let number = stable_sensitive_number("id-card", original);
        let month = number % 12 + 1;
        let day = (number / 12) % 28 + 1;
        let seq = (number / (12 * 28)) % 1000;
        self.unique_pseudonym(format!("1101011990{month:02}{day:02}{seq:03}X"))
    }

    fn stable_bank_card(&mut self, original: &str) -> String {
        let number = stable_sensitive_number("bank-card", original) % 10_000_000_000_000;
        self.unique_pseudonym(format!("622202{number:013}"))
    }

    fn unique_pseudonym(&mut self, pseudonym: String) -> String {
        if self.used_pseudonyms.insert(pseudonym.clone()) {
            return pseudonym;
        }

        let mut suffix = 1usize;
        loop {
            let candidate = format!("{pseudonym}{suffix}");
            if self.used_pseudonyms.insert(candidate.clone()) {
                return candidate;
            }
            suffix += 1;
        }
    }
}

fn stable_sensitive_hash(kind: &str, original: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update([0]);
    hasher.update(original.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn stable_sensitive_number(kind: &str, original: &str) -> u64 {
    let hash = stable_sensitive_hash(kind, original);
    u64::from_str_radix(&hash[..16], 16).unwrap_or(0)
}

fn redacted_value(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(format!("<redacted,len={}>", text.chars().count())),
        Value::Array(_) | Value::Object(_) => Value::String(REDACTED.to_string()),
        Value::Null => Value::Null,
        _ => Value::String(REDACTED.to_string()),
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|ch| *ch != '-' && *ch != '_')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "apikey"
            | "authorization"
            | "cookie"
            | "joycodecookie"
            | "ptkey"
            | "xptkey"
            | "token"
            | "accesstoken"
            | "refreshtoken"
            | "idtoken"
            | "metoken"
            | "iamtoken"
            | "xjacptoken"
            | "ssaticket"
            | "chatid"
            | "userid"
            | "uid"
            | "username"
            | "email"
            | "mobile"
            | "phone"
            | "idcard"
            | "identitycard"
            | "bankcard"
            | "cardno"
            | "cardnumber"
            | "account"
    )
}

fn is_credential_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|ch| *ch != '-' && *ch != '_')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "apikey"
            | "authorization"
            | "cookie"
            | "joycodecookie"
            | "ptkey"
            | "xptkey"
            | "token"
            | "accesstoken"
            | "refreshtoken"
            | "idtoken"
            | "metoken"
            | "iamtoken"
            | "xjacptoken"
            | "ssaticket"
    )
}

fn json_string_field_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r#"(?i)("((?:api[_-]?key|authorization|cookie|joycode[_-]?cookie|pt[_-]?key|x[_-]?pt[_-]?key|token|access[_-]?token|refresh[_-]?token|id[_-]?token|me[_-]?token|iam[_-]?token|x[_-]?jacp[_-]?token|ssa[_-]?ticket|chat[_-]?id|user[_-]?id|uid|username|email|mobile|phone|account))"\s*:\s*")([^"]*)(")"#,
        )
        .expect("valid sensitive json regex")
    })
}

fn cookie_pair_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r#"(?i)(^|[;\s])([A-Za-z0-9_.-]*(?:token|ticket|key|pin|uid|erp|email|mobile|phone)[A-Za-z0-9_.-]*)=([^;\s"]+)"#,
        )
        .expect("valid cookie regex")
    })
}

fn email_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#).expect("valid email regex")
    })
}

fn phone_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(?x)(^|[^0-9A-Za-z])((?:\+?86[-\s]?)?1[3-9]\d[-\s]?\d{4}[-\s]?\d{4})($|[^0-9A-Za-z])"#)
            .expect("valid phone regex")
    })
}

fn cn_id_card_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(?i)(^|[^0-9A-Za-z])([1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dX])($|[^0-9A-Za-z])"#)
            .expect("valid id card regex")
    })
}

fn bank_card_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(^|[^0-9A-Za-z])((?:\d[ -]?){12,18}\d)($|[^0-9A-Za-z])"#)
            .expect("valid bank card regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_sensitive_json_text() {
        let text = r#"{"apiKey":"sk-secret","token":"mt_secret","chatId":"abc","email":"u@example.com","mobile":"13476650547","text":"身份证11010119900307777X 银行卡6222020202020202020"}"#;
        let redacted = redact_sensitive_text(text);
        assert!(!redacted.contains("sk-secret"));
        assert!(!redacted.contains("mt_secret"));
        assert!(!redacted.contains("abc"));
        assert!(!redacted.contains("u@example.com"));
        assert!(!redacted.contains("13476650547"));
        assert!(!redacted.contains("11010119900307777X"));
        assert!(!redacted.contains("6222020202020202020"));
        assert!(redacted.contains("<redacted"));
    }

    #[test]
    fn redacts_sensitive_value_recursively() {
        let mut value = json!({
            "model": "safe-model",
            "apiKey": "sk-secret",
            "messages": [{ "content": "手机号13476650547 身份证11010119900307777X 银行卡6222020202020202020" }],
            "routes": [{ "headers": { "cookie": "pt_key=secret; pt_pin=user" } }]
        });
        redact_sensitive_value(&mut value);
        assert_eq!(value["model"], "safe-model");
        assert_ne!(value["apiKey"], "sk-secret");
        assert_ne!(
            value["routes"][0]["headers"]["cookie"],
            "pt_key=secret; pt_pin=user"
        );
        let content = value["messages"][0]["content"].as_str().unwrap();
        assert!(!content.contains("13476650547"));
        assert!(!content.contains("11010119900307777X"));
        assert!(!content.contains("6222020202020202020"));
    }

    #[test]
    fn pseudonymizes_and_restores_sensitive_text_values() {
        let mut value = json!({
            "model": "safe-model",
            "messages": [{
                "content": "邮箱u@example.com 手机13476650547 身份证11010119900307777X 银行卡6222020202020202020"
            }]
        });
        let map = pseudonymize_sensitive_value(&mut value);
        let content = value["messages"][0]["content"].as_str().unwrap();
        assert!(!content.contains("u@example.com"));
        assert!(!content.contains("13476650547"));
        assert!(!content.contains("11010119900307777X"));
        assert!(!content.contains("6222020202020202020"));
        assert!(content.contains("example.invalid"));
        assert!(content.contains("139"));
        assert!(content.contains("1101011990"));
        assert!(content.contains("622202"));

        let restored = map.restore_text(content);
        assert!(restored.contains("u@example.com"));
        assert!(restored.contains("13476650547"));
        assert!(restored.contains("11010119900307777X"));
        assert!(restored.contains("6222020202020202020"));
    }

    #[test]
    fn pseudonymizer_redacts_credentials_without_mapping_them() {
        let mut value = json!({
            "apiKey": "sk-secret",
            "messages": [{ "content": "手机号13476650547" }]
        });
        let map = pseudonymize_sensitive_value(&mut value);
        assert_ne!(value["apiKey"], "sk-secret");
        assert_eq!(map.restore_text("sk-secret"), "sk-secret");
    }

    #[test]
    fn pseudonymizer_is_stable_across_context_order() {
        let mut first = json!({
            "system": "文件 FETCH_HEAD.sync-conflict-6222020202020202020-PXVHV5K",
            "messages": [{ "content": "手机号13476650547" }]
        });
        let mut second = json!({
            "messages": [{ "content": "身份证11010119900307777X 邮箱u@example.com 手机13476650547" }],
            "system": "文件 FETCH_HEAD.sync-conflict-6222020202020202020-PXVHV5K"
        });
        pseudonymize_sensitive_value(&mut first);
        pseudonymize_sensitive_value(&mut second);
        assert_eq!(first["system"], second["system"]);
    }

    #[test]
    fn pseudonymizer_preserves_tool_json_schema_properties() {
        let mut value = json!({
            "tools": [{
                "name": "mcp1_browser_press_key",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "token": { "type": "string" }
                    },
                    "required": ["key"]
                }
            }]
        });
        pseudonymize_sensitive_value(&mut value);
        assert_eq!(
            value["tools"][0]["input_schema"]["properties"]["key"]["type"],
            "string"
        );
        assert_eq!(
            value["tools"][0]["input_schema"]["properties"]["token"]["type"],
            "string"
        );
    }
}

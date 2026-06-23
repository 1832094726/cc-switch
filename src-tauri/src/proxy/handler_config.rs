//! Handler 配置模块
//!
//! 定义各 API 处理器的配置结构和使用量解析器

use crate::app_config::AppType;
use crate::proxy::usage::parser::TokenUsage;
use serde_json::Value;

/// 使用量解析器类型别名
pub type StreamUsageParser = fn(&[Value]) -> Option<TokenUsage>;
pub type ResponseUsageParser = fn(&Value) -> Option<TokenUsage>;

/// 模型提取器类型别名
/// 参数: (流式事件列表, 请求中的模型名称) -> 最终使用的模型名称
pub type StreamModelExtractor = fn(&[Value], &str) -> String;

/// 流式 usage 事件预过滤器类型别名。
///
/// 参数是 SSE `data:` 原始字符串。返回 false 时跳过 JSON parse，避免在
/// token/chunk 高频路径上解析与 usage 无关的事件。
pub type StreamUsageEventFilter = fn(&str) -> bool;

/// 各 API 的使用量解析配置
#[derive(Clone, Copy)]
pub struct UsageParserConfig {
    /// 流式响应解析器
    pub stream_parser: StreamUsageParser,
    /// 非流式响应解析器
    pub response_parser: ResponseUsageParser,
    /// 流式响应中的模型提取器
    pub model_extractor: StreamModelExtractor,
    /// 流式 usage 事件预过滤器
    pub stream_event_filter: Option<StreamUsageEventFilter>,
    /// 应用类型字符串（用于日志记录）
    pub app_type_str: &'static str,
}

// ============================================================================
// 流式 usage 事件预过滤
// ============================================================================

pub fn claude_stream_usage_event_filter(data: &str) -> bool {
    data.contains("message_start")
        || data.contains("message_delta")
        || data.contains("input_tokens")
        || data.contains("output_tokens")
        || data.contains("cache_read_input_tokens")
        || data.contains("cache_creation_input_tokens")
}

fn openai_stream_usage_event_filter(data: &str) -> bool {
    data.contains("\"usage\"")
}

pub fn codex_stream_usage_event_filter(data: &str) -> bool {
    data.contains("\"response.completed\"") || data.contains("\"usage\"")
}

pub fn devin_stream_usage_event_filter(data: &str) -> bool {
    claude_stream_usage_event_filter(data) || codex_stream_usage_event_filter(data)
}

fn gemini_stream_usage_event_filter(data: &str) -> bool {
    data.contains("\"usageMetadata\"")
}

// ============================================================================
// 模型提取器实现
// ============================================================================

/// Claude 流式响应模型提取（优先使用 usage.model）
///
/// 空字符串模型名视为缺失（转换层对无回显上游会合成 model:""），
/// 落到 fallback_model（映射后的出站模型或客户端请求模型）。
fn claude_model_extractor(events: &[Value], fallback_model: &str) -> String {
    // 首先尝试从解析的 usage 中获取模型
    if let Some(usage) = TokenUsage::from_claude_stream_events(events) {
        if let Some(model) = usage.model.filter(|m| !m.is_empty()) {
            return model;
        }
    }
    fallback_model.to_string()
}

/// OpenAI Chat Completions 流式响应模型提取（优先使用 usage.model）
fn openai_model_extractor(events: &[Value], fallback_model: &str) -> String {
    // 首先尝试从解析的 usage 中获取模型
    if let Some(usage) = TokenUsage::from_openai_stream_events(events) {
        if let Some(model) = usage.model.filter(|m| !m.is_empty()) {
            return model;
        }
    }
    // 回退：从事件中直接提取
    events
        .iter()
        .find_map(|e| e.get("model")?.as_str().filter(|m| !m.is_empty()))
        .unwrap_or(fallback_model)
        .to_string()
}

/// Codex 智能流式响应模型提取（自动检测格式）
fn codex_auto_model_extractor(events: &[Value], fallback_model: &str) -> String {
    // 首先尝试从解析的 usage 中获取模型
    if let Some(usage) = TokenUsage::from_codex_stream_events_auto(events) {
        if let Some(model) = usage.model.filter(|m| !m.is_empty()) {
            return model;
        }
    }
    // 回退：从 response.completed 事件中提取
    events
        .iter()
        .find_map(|e| {
            if e.get("type")?.as_str()? == "response.completed" {
                e.get("response")?
                    .get("model")?
                    .as_str()
                    .filter(|m| !m.is_empty())
            } else {
                None
            }
        })
        .or_else(|| {
            // 再回退：从 OpenAI 格式事件中提取
            events
                .iter()
                .find_map(|e| e.get("model")?.as_str().filter(|m| !m.is_empty()))
        })
        .unwrap_or(fallback_model)
        .to_string()
}

fn devin_model_extractor(events: &[Value], fallback_model: &str) -> String {
    if let Some(usage) = TokenUsage::from_claude_stream_events(events)
        .or_else(|| TokenUsage::from_codex_stream_events_auto(events))
    {
        if let Some(model) = usage.model.filter(|m| !m.is_empty()) {
            return model;
        }
    }

    events
        .iter()
        .find_map(|event| {
            event
                .get("model")
                .or_else(|| event.pointer("/response/model"))?
                .as_str()
                .filter(|m| !m.is_empty())
        })
        .unwrap_or(fallback_model)
        .to_string()
}

/// Gemini 流式响应模型提取（优先使用 usage.model）
fn gemini_model_extractor(events: &[Value], fallback_model: &str) -> String {
    // 首先尝试从解析的 usage 中获取模型
    if let Some(usage) = TokenUsage::from_gemini_stream_chunks(events) {
        if let Some(model) = usage.model.filter(|m| !m.is_empty()) {
            return model;
        }
    }
    fallback_model.to_string()
}

// ============================================================================
// 预定义配置
// ============================================================================

/// Claude API 解析配置
pub const CLAUDE_PARSER_CONFIG: UsageParserConfig = UsageParserConfig {
    stream_parser: TokenUsage::from_claude_stream_events,
    response_parser: TokenUsage::from_claude_response,
    model_extractor: claude_model_extractor,
    stream_event_filter: Some(claude_stream_usage_event_filter),
    app_type_str: "claude",
};

/// OpenAI Chat Completions API 解析配置（用于 Codex /v1/chat/completions）
pub const OPENAI_PARSER_CONFIG: UsageParserConfig = UsageParserConfig {
    stream_parser: TokenUsage::from_openai_stream_events,
    response_parser: TokenUsage::from_openai_response,
    model_extractor: openai_model_extractor,
    stream_event_filter: Some(openai_stream_usage_event_filter),
    app_type_str: "codex",
};

/// Codex 智能解析配置（自动检测 OpenAI 或 Codex 格式）
pub const CODEX_PARSER_CONFIG: UsageParserConfig = UsageParserConfig {
    stream_parser: TokenUsage::from_codex_stream_events_auto,
    response_parser: TokenUsage::from_codex_response_auto,
    model_extractor: codex_auto_model_extractor,
    stream_event_filter: Some(codex_stream_usage_event_filter),
    app_type_str: "codex",
};

fn parse_devin_stream_usage(events: &[Value]) -> Option<TokenUsage> {
    TokenUsage::from_claude_stream_events(events)
        .or_else(|| TokenUsage::from_codex_stream_events_auto(events))
}

fn parse_devin_response_usage(body: &Value) -> Option<TokenUsage> {
    TokenUsage::from_claude_response(body).or_else(|| TokenUsage::from_codex_response_auto(body))
}

/// Devin/Windsurf can be routed to Anthropic Messages, OpenAI Chat, or Responses.
pub const DEVIN_PARSER_CONFIG: UsageParserConfig = UsageParserConfig {
    stream_parser: parse_devin_stream_usage,
    response_parser: parse_devin_response_usage,
    model_extractor: devin_model_extractor,
    stream_event_filter: Some(devin_stream_usage_event_filter),
    app_type_str: "devin",
};

/// Gemini API 解析配置
pub const GEMINI_PARSER_CONFIG: UsageParserConfig = UsageParserConfig {
    stream_parser: TokenUsage::from_gemini_stream_chunks,
    response_parser: TokenUsage::from_gemini_response,
    model_extractor: gemini_model_extractor,
    stream_event_filter: Some(gemini_stream_usage_event_filter),
    app_type_str: "gemini",
};

// ============================================================================
// Handler 配置（预留，用于进一步简化）
// ============================================================================

/// Handler 基础配置
///
/// 预留结构，可用于进一步统一各 handler 的配置
#[allow(dead_code)]
#[derive(Clone)]
pub struct HandlerConfig {
    /// 应用类型
    pub app_type: AppType,
    /// 日志标签
    pub tag: &'static str,
    /// 应用类型字符串
    pub app_type_str: &'static str,
    /// 使用量解析配置
    pub parser_config: &'static UsageParserConfig,
}

/// Claude Handler 配置
#[allow(dead_code)]
pub const CLAUDE_HANDLER_CONFIG: HandlerConfig = HandlerConfig {
    app_type: AppType::Claude,
    tag: "Claude",
    app_type_str: "claude",
    parser_config: &CLAUDE_PARSER_CONFIG,
};

/// Codex Chat Completions Handler 配置
#[allow(dead_code)]
pub const CODEX_CHAT_HANDLER_CONFIG: HandlerConfig = HandlerConfig {
    app_type: AppType::Codex,
    tag: "Codex",
    app_type_str: "codex",
    parser_config: &OPENAI_PARSER_CONFIG,
};

/// Codex Responses Handler 配置
#[allow(dead_code)]
pub const CODEX_RESPONSES_HANDLER_CONFIG: HandlerConfig = HandlerConfig {
    app_type: AppType::Codex,
    tag: "Codex",
    app_type_str: "codex",
    parser_config: &CODEX_PARSER_CONFIG,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn devin_parser_reads_openai_chat_usage() {
        let body = json!({
            "model": "Claude-Sonnet-4.6-hq",
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 5,
                "total_tokens": 13
            }
        });

        let usage = (DEVIN_PARSER_CONFIG.response_parser)(&body).expect("usage");

        assert_eq!(usage.input_tokens, 8);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.model.as_deref(), Some("Claude-Sonnet-4.6-hq"));
    }

    #[test]
    fn devin_parser_reads_claude_stream_usage() {
        let events = vec![
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "model": "claude-sonnet-4-6",
                    "usage": {
                        "input_tokens": 11,
                        "cache_read_input_tokens": 3
                    }
                }
            }),
            json!({
                "type": "message_delta",
                "usage": {
                    "output_tokens": 7
                }
            }),
        ];

        let usage = (DEVIN_PARSER_CONFIG.stream_parser)(&events).expect("usage");
        let model = (DEVIN_PARSER_CONFIG.model_extractor)(&events, "fallback");

        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_read_tokens, 3);
        assert_eq!(model, "claude-sonnet-4-6");
    }

    #[test]
    fn devin_filter_collects_joycode_claude_usage_delta() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":63967,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":592}}"#;
        assert!(devin_stream_usage_event_filter(data));

        let event = serde_json::from_str::<serde_json::Value>(data).unwrap();
        let usage = (DEVIN_PARSER_CONFIG.stream_parser)(&[event]).expect("usage");
        assert_eq!(usage.input_tokens, 63967);
        assert_eq!(usage.output_tokens, 592);
        assert_eq!(usage.cache_creation_tokens, 0);
        assert_eq!(usage.cache_read_tokens, 0);
    }
}

/// Gemini Handler 配置
#[allow(dead_code)]
pub const GEMINI_HANDLER_CONFIG: HandlerConfig = HandlerConfig {
    app_type: AppType::Gemini,
    tag: "Gemini",
    app_type_str: "gemini",
    parser_config: &GEMINI_PARSER_CONFIG,
};

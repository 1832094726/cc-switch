//! Windsurf Connect-RPC protobuf bridge for Devin.
//!
//! Devin speaks Windsurf's `GetChatMessage` Connect/protobuf method.  This
//! module keeps that wire protocol at the cc-switch boundary: request protobuf
//! is decoded into the same Anthropic-like canonical shape used by
//! windsurf-proxy, and upstream SSE is encoded back into Windsurf protobuf
//! frames.

use super::{
    forwarder::ActiveConnectionGuard,
    sse::{append_utf8_safe, strip_sse_field, take_sse_block},
    ProxyError,
};
use axum::http::{HeaderMap, HeaderValue};
use bytes::Bytes;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Instant;

const SOURCE_UNSPECIFIED: u64 = 0;
const SOURCE_USER: u64 = 1;
const SOURCE_SYSTEM: u64 = 2;
const SOURCE_UNKNOWN: u64 = 3;
const SOURCE_TOOL: u64 = 4;
const SOURCE_SYSTEM_PROMPT: u64 = 5;

const STOP_PATTERN: u64 = 2;
const STOP_MAX_TOKENS: u64 = 3;
const STOP_FUNCTION_CALL: u64 = 10;
const STOP_ERROR: u64 = 13;
const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 12_000;
const DEFAULT_MAX_TOKENS: u64 = 16_384;
const SMALL_MODEL_MAX_TOKENS: u64 = 768;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpstreamWireApi {
    AnthropicMessages,
    OpenAIResponses,
    OpenAIChatCompletions,
}

impl UpstreamWireApi {
    pub(crate) fn from_endpoint(endpoint: &str) -> Self {
        let path = endpoint.split_once('?').map_or(endpoint, |(path, _)| path);
        if path == "/messages" || path.ends_with("/messages") {
            Self::AnthropicMessages
        } else if path == "/responses"
            || path.ends_with("/responses")
            || path == "/responses/compact"
            || path.ends_with("/responses/compact")
        {
            Self::OpenAIResponses
        } else {
            Self::OpenAIChatCompletions
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WindsurfChatRequest {
    pub(crate) system_prompt: String,
    pub(crate) messages: Vec<Value>,
    pub(crate) tools: Vec<Value>,
    pub(crate) tool_choice: Option<Value>,
    pub(crate) requested_model: Option<String>,
    pub(crate) initiator: String,
}

impl WindsurfChatRequest {
    pub(crate) fn to_anthropic_body(&self) -> Value {
        let is_small_model = self
            .requested_model
            .as_deref()
            .is_some_and(is_devin_small_model_alias);
        let max_tokens = if is_small_model {
            SMALL_MODEL_MAX_TOKENS
        } else {
            DEFAULT_MAX_TOKENS
        };
        let mut body = json!({
            "_cc_switch_canonical_api": "anthropic_messages",
            "messages": self.messages,
            "stream": true,
            "max_tokens": max_tokens,
            "temperature": 0
        });

        if !is_small_model {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": DEFAULT_THINKING_BUDGET_TOKENS
            });
        }

        if let Some(model) = self
            .requested_model
            .as_deref()
            .filter(|model| !model.is_empty())
        {
            body["model"] = json!(model);
        }
        if !self.system_prompt.trim().is_empty() {
            body["system"] = json!(self.system_prompt);
        }
        if !self.tools.is_empty() {
            body["tools"] = Value::Array(self.tools.clone());
        }
        if let Some(tool_choice) = &self.tool_choice {
            body["tool_choice"] = tool_choice.clone();
        }

        body
    }
}

pub fn is_devin_small_model_alias(model: &str) -> bool {
    matches!(
        model.trim(),
        "MODEL_GPT_5_NANO" | "MODEL_GOOGLE_GEMINI_2_5_FLASH" | "MODEL_CHAT_GPT_4_1_MINI_2025_04_14"
    )
}

#[derive(Debug, Clone)]
struct ParsedPrompt {
    source: u64,
    prompt: String,
    tool_calls: Vec<ParsedToolCall>,
    tool_call_id: String,
    tool_result_is_error: bool,
    images: Vec<ParsedImage>,
    thinking: String,
    signature: String,
}

#[derive(Debug, Clone)]
struct ParsedToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

#[derive(Debug, Clone)]
struct ParsedImage {
    base64_data: String,
    mime_type: String,
}

#[derive(Debug, Clone)]
struct ProtoField {
    field: u32,
    value: ProtoValue,
}

#[derive(Debug, Clone)]
enum ProtoValue {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(Vec<u8>),
    Fixed32([u8; 4]),
}

pub(crate) fn stream_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/connect+proto"),
    );
    headers.insert("connect-content-encoding", HeaderValue::from_static("gzip"));
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache"),
    );
    headers
}

pub(crate) fn error_response_body(message_id: &str, error_text: &str) -> Result<Bytes, ProxyError> {
    let mut body = Vec::new();
    body.extend_from_slice(&wrap_envelope(&build_error_chunk(message_id, error_text))?);
    body.extend_from_slice(&end_of_stream_envelope()?);
    Ok(Bytes::from(body))
}

pub(crate) fn parse_get_chat_message_request(
    body: &[u8],
    headers: &HeaderMap,
) -> Result<WindsurfChatRequest, ProxyError> {
    let proto = unwrap_connect_request(body, headers)?;
    let fields = parse_fields(&proto)?;

    let mut system_prompt = get_field_string(&fields, 2).unwrap_or_default();
    let requested_model = get_field_string(&fields, 21)
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty());

    let parsed_prompts: Vec<ParsedPrompt> = get_all_field_bytes(&fields, 3)
        .into_iter()
        .map(parse_chat_message_prompt)
        .collect::<Result<_, _>>()?;

    for prompt in &parsed_prompts {
        if prompt.source == SOURCE_SYSTEM_PROMPT && !prompt.prompt.is_empty() {
            if system_prompt.is_empty() {
                system_prompt = prompt.prompt.clone();
            } else {
                system_prompt.push('\n');
                system_prompt.push_str(&prompt.prompt);
            }
        }
    }

    let initiator = infer_initiator(&parsed_prompts);
    let messages = merge_consecutive_messages(
        parsed_prompts
            .iter()
            .filter_map(prompt_to_anthropic_message)
            .collect(),
    );

    let tools = get_all_field_bytes(&fields, 10)
        .into_iter()
        .map(parse_chat_tool_definition)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|tool| {
            tool.get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty())
        })
        .collect::<Vec<_>>();

    let tool_choice = get_field_bytes(&fields, 12)
        .map(parse_chat_tool_choice)
        .transpose()?
        .flatten();

    Ok(WindsurfChatRequest {
        system_prompt,
        messages,
        tools,
        tool_choice,
        requested_model,
        initiator,
    })
}

pub(crate) fn rewrite_register_user_response_body(
    body: &[u8],
    api_server_url: &str,
) -> Result<Option<Vec<u8>>, ProxyError> {
    if let Some(rewritten) = rewrite_register_user_connect_envelope(body, api_server_url)? {
        return Ok(Some(rewritten));
    }

    if let Ok(decoded) = gunzip_bytes(body) {
        if let Some(rewritten_proto) = rewrite_register_user_proto(&decoded, api_server_url)? {
            return Ok(Some(rewritten_proto));
        }
    }

    rewrite_register_user_proto(body, api_server_url)
}

fn rewrite_register_user_connect_envelope(
    body: &[u8],
    api_server_url: &str,
) -> Result<Option<Vec<u8>>, ProxyError> {
    if body.len() < 5 {
        return Ok(None);
    }

    let flags = body[0];
    if flags > 1 {
        return Ok(None);
    }

    let msg_len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    if msg_len != body.len().saturating_sub(5) {
        return Ok(None);
    }

    let payload = &body[5..];
    let proto = if flags == 1 {
        gunzip_bytes(payload).map_err(|e| {
            ProxyError::InvalidRequest(format!("Invalid RegisterUser gzip envelope: {e}"))
        })?
    } else {
        payload.to_vec()
    };

    let Some(rewritten_proto) = rewrite_register_user_proto(&proto, api_server_url)? else {
        return Ok(None);
    };

    let mut out = Vec::with_capacity(5 + rewritten_proto.len());
    out.push(0);
    out.extend_from_slice(&(rewritten_proto.len() as u32).to_be_bytes());
    out.extend_from_slice(&rewritten_proto);
    Ok(Some(out))
}

fn rewrite_register_user_proto(
    proto: &[u8],
    api_server_url: &str,
) -> Result<Option<Vec<u8>>, ProxyError> {
    let fields = parse_fields(proto)?;
    let mut replaced = false;
    let mut out = Vec::new();

    for field in fields {
        match field.value {
            ProtoValue::Varint(value) => out.extend(write_varint_field(field.field, value)),
            ProtoValue::Fixed64(value) => out.extend(write_fixed64_field(field.field, value)),
            ProtoValue::Bytes(value) => {
                if field.field == 3 {
                    replaced = true;
                    out.extend(write_string_field(3, api_server_url));
                } else {
                    out.extend(write_bytes_field(field.field, &value));
                }
            }
            ProtoValue::Fixed32(value) => out.extend(write_fixed32_field(field.field, value)),
        }
    }

    Ok(replaced.then_some(out))
}

pub(crate) fn connect_stream_from_sse<S>(
    upstream: S,
    wire_api: UpstreamWireApi,
    message_id: String,
    model_uid: String,
    connection_guard: Option<ActiveConnectionGuard>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    async_stream::stream! {
        let _guard = connection_guard;
        let started_at = Instant::now();
        let mut upstream = Box::pin(upstream);
        let mut text_buffer = String::new();
        let mut utf8_remainder = Vec::new();
        let mut processor = StreamProcessor::new(wire_api, message_id, model_uid);

        while let Some(item) = upstream.next().await {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(err) => {
                    log::warn!("[Devin/Windsurf] Upstream stream error: {err}");
                    let error = build_error_chunk(processor.message_id(), "[Stream Error]");
                    yield wrap_envelope_io(&error);
                    yield end_of_stream_envelope_io();
                    return;
                }
            };

            append_utf8_safe(&mut text_buffer, &mut utf8_remainder, &chunk);
            while let Some(block) = take_sse_block(&mut text_buffer) {
                for frame in processor.process_sse_block(&block, started_at.elapsed().as_millis() as f64) {
                    yield wrap_envelope_io(&frame);
                }
                if processor.is_done() {
                    yield end_of_stream_envelope_io();
                    return;
                }
            }
        }

        if !text_buffer.trim().is_empty() {
            let tail = std::mem::take(&mut text_buffer);
            for frame in processor.process_sse_block(&tail, started_at.elapsed().as_millis() as f64) {
                yield wrap_envelope_io(&frame);
            }
        }

        if !processor.is_done() {
            for frame in processor.force_done(started_at.elapsed().as_millis() as f64) {
                yield wrap_envelope_io(&frame);
            }
        }
        yield end_of_stream_envelope_io();
    }
}

fn unwrap_connect_request(body: &[u8], headers: &HeaderMap) -> Result<Vec<u8>, ProxyError> {
    let mut buf = body.to_vec();
    let content_encoding = headers
        .get("connect-content-encoding")
        .or_else(|| headers.get(axum::http::header::CONTENT_ENCODING))
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_encoding.contains("gzip") {
        if let Ok(decoded) = gunzip_bytes(&buf) {
            buf = decoded;
        }
    }

    if buf.len() >= 5 {
        let flags = buf[0];
        let msg_len = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
        if msg_len == buf.len().saturating_sub(5) && flags <= 1 {
            let payload = &buf[5..];
            return if flags == 1 {
                gunzip_bytes(payload).map_err(|e| {
                    ProxyError::InvalidRequest(format!("Invalid Windsurf gzip envelope: {e}"))
                })
            } else {
                Ok(payload.to_vec())
            };
        }
    }

    Ok(buf)
}

fn parse_chat_message_prompt(buf: &[u8]) -> Result<ParsedPrompt, ProxyError> {
    let fields = parse_fields(buf)?;
    let tool_calls = get_all_field_bytes(&fields, 6)
        .into_iter()
        .map(parse_chat_tool_call)
        .collect::<Result<Vec<_>, _>>()?;
    let images = get_all_field_bytes(&fields, 10)
        .into_iter()
        .map(parse_image_data)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ParsedPrompt {
        source: get_field_varint(&fields, 2).unwrap_or(SOURCE_UNSPECIFIED),
        prompt: get_field_string(&fields, 3).unwrap_or_default(),
        tool_calls,
        tool_call_id: get_field_string(&fields, 7).unwrap_or_default(),
        tool_result_is_error: get_field_varint(&fields, 9).unwrap_or(0) != 0,
        images,
        thinking: get_field_string(&fields, 11).unwrap_or_default(),
        signature: get_field_string(&fields, 12).unwrap_or_default(),
    })
}

fn parse_image_data(buf: &[u8]) -> Result<ParsedImage, ProxyError> {
    let fields = parse_fields(buf)?;
    Ok(ParsedImage {
        base64_data: get_field_string(&fields, 1).unwrap_or_default(),
        mime_type: get_field_string(&fields, 2).unwrap_or_else(|| "image/png".to_string()),
    })
}

fn parse_chat_tool_call(buf: &[u8]) -> Result<ParsedToolCall, ProxyError> {
    let fields = parse_fields(buf)?;
    Ok(ParsedToolCall {
        id: get_field_string(&fields, 1).unwrap_or_default(),
        name: get_field_string(&fields, 2).unwrap_or_default(),
        arguments_json: get_field_string(&fields, 3).unwrap_or_else(|| "{}".to_string()),
    })
}

fn parse_chat_tool_definition(buf: &[u8]) -> Result<Value, ProxyError> {
    let fields = parse_fields(buf)?;
    let schema = get_field_string(&fields, 3).unwrap_or_else(|| "{}".to_string());
    let input_schema = serde_json::from_str::<Value>(&schema)
        .unwrap_or_else(|_| json!({ "type": "object", "properties": {} }));

    Ok(json!({
        "name": get_field_string(&fields, 1).unwrap_or_default(),
        "description": get_field_string(&fields, 2).unwrap_or_default(),
        "input_schema": input_schema
    }))
}

fn parse_chat_tool_choice(buf: &[u8]) -> Result<Option<Value>, ProxyError> {
    let fields = parse_fields(buf)?;
    let tool_name = get_field_string(&fields, 2).unwrap_or_default();
    if !tool_name.trim().is_empty() {
        return Ok(Some(json!({ "type": "tool", "name": tool_name.trim() })));
    }

    let option_name = get_field_string(&fields, 1).unwrap_or_default();
    if !option_name.trim().is_empty() {
        return Ok(Some(json!({ "type": option_name.trim() })));
    }

    Ok(None)
}

fn prompt_to_anthropic_message(prompt: &ParsedPrompt) -> Option<Value> {
    match prompt.source {
        SOURCE_UNSPECIFIED | SOURCE_SYSTEM_PROMPT => None,
        SOURCE_TOOL => {
            let mut block = json!({
                "type": "tool_result",
                "tool_use_id": prompt.tool_call_id,
                "content": prompt.prompt
            });
            if prompt.tool_result_is_error {
                block["is_error"] = json!(true);
            }
            Some(json!({ "role": "user", "content": [block] }))
        }
        SOURCE_USER => {
            if prompt.images.is_empty() {
                return Some(json!({ "role": "user", "content": prompt.prompt }));
            }
            let mut content = Vec::new();
            for image in &prompt.images {
                if image.base64_data.is_empty() {
                    continue;
                }
                content.push(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.mime_type,
                        "data": image.base64_data
                    }
                }));
            }
            if !prompt.prompt.is_empty() {
                content.push(json!({ "type": "text", "text": prompt.prompt }));
            }
            Some(json!({ "role": "user", "content": content }))
        }
        SOURCE_SYSTEM | SOURCE_UNKNOWN => {
            let mut content = Vec::new();
            if !prompt.thinking.is_empty() {
                if !prompt.signature.is_empty() {
                    content.push(json!({
                        "type": "thinking",
                        "thinking": prompt.thinking,
                        "signature": prompt.signature
                    }));
                } else {
                    // Without a signature we cannot reconstruct a valid
                    // thinking block. Drop it silently rather than wrapping
                    // it in a visible <cc-switch:thinking> text tag that
                    // would leak into the conversation.
                }
            }
            if !prompt.prompt.is_empty() {
                content.push(json!({ "type": "text", "text": prompt.prompt }));
            }
            for tool_call in &prompt.tool_calls {
                let input = serde_json::from_str::<Value>(&tool_call.arguments_json)
                    .unwrap_or_else(|_| json!({}));
                content.push(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": input
                }));
            }

            if content.len() == 1 && content[0].get("type").and_then(Value::as_str) == Some("text")
            {
                return Some(json!({
                    "role": "assistant",
                    "content": content[0].get("text").cloned().unwrap_or(Value::String(String::new()))
                }));
            }

            Some(json!({ "role": "assistant", "content": content }))
        }
        _ => None,
    }
}

fn infer_initiator(prompts: &[ParsedPrompt]) -> String {
    let non_system = prompts
        .iter()
        .filter(|prompt| !matches!(prompt.source, SOURCE_SYSTEM_PROMPT | SOURCE_UNSPECIFIED))
        .collect::<Vec<_>>();
    let Some(last) = non_system.last() else {
        return "agent".to_string();
    };

    if last.source == SOURCE_USER {
        let second_to_last = non_system.get(non_system.len().saturating_sub(2));
        if !second_to_last.is_some_and(|prompt| prompt.source == SOURCE_TOOL) {
            return "user".to_string();
        }
    }

    "agent".to_string()
}

fn merge_consecutive_messages(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();

    for mut message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        if let Some(previous) = merged.last_mut().filter(|previous| {
            previous
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|previous_role| previous_role == role)
        }) {
            let mut blocks = content_to_blocks(previous.get("content"));
            blocks.extend(content_to_blocks(message.get("content")));
            previous["content"] = Value::Array(blocks);
        } else {
            merged.push(std::mem::take(&mut message));
        }
    }

    for message in &mut merged {
        if let Some(blocks) = message.get("content").and_then(Value::as_array) {
            if blocks.len() == 1 && blocks[0].get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = blocks[0].get("text").cloned() {
                    message["content"] = text;
                }
            }
        }
    }

    merged
}

fn content_to_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => vec![json!({ "type": "text", "text": text })],
        Some(Value::Array(blocks)) => blocks.clone(),
        Some(value) if !value.is_null() => {
            vec![json!({ "type": "text", "text": value.to_string() })]
        }
        _ => Vec::new(),
    }
}

fn parse_fields(buf: &[u8]) -> Result<Vec<ProtoField>, ProxyError> {
    let mut fields = Vec::new();
    let mut offset = 0usize;

    while offset < buf.len() {
        let tag = decode_varint(buf, &mut offset)?;
        let field = (tag >> 3) as u32;
        let wire_type = (tag & 7) as u8;
        if field == 0 {
            break;
        }

        let value = match wire_type {
            0 => ProtoValue::Varint(decode_varint(buf, &mut offset)?),
            1 => {
                if offset + 8 > buf.len() {
                    return Err(ProxyError::InvalidRequest(
                        "Truncated protobuf fixed64 field".to_string(),
                    ));
                }
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&buf[offset..offset + 8]);
                offset += 8;
                ProtoValue::Fixed64(bytes)
            }
            2 => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len > buf.len() {
                    return Err(ProxyError::InvalidRequest(
                        "Truncated protobuf bytes field".to_string(),
                    ));
                }
                let bytes = buf[offset..offset + len].to_vec();
                offset += len;
                ProtoValue::Bytes(bytes)
            }
            5 => {
                if offset + 4 > buf.len() {
                    return Err(ProxyError::InvalidRequest(
                        "Truncated protobuf fixed32 field".to_string(),
                    ));
                }
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&buf[offset..offset + 4]);
                offset += 4;
                ProtoValue::Fixed32(bytes)
            }
            _ => {
                return Err(ProxyError::InvalidRequest(format!(
                    "Unsupported protobuf wire type: {wire_type}"
                )))
            }
        };

        fields.push(ProtoField { field, value });
    }

    Ok(fields)
}

fn decode_varint(buf: &[u8], offset: &mut usize) -> Result<u64, ProxyError> {
    let mut result = 0u64;
    let mut shift = 0u32;

    while *offset < buf.len() {
        let byte = buf[*offset];
        *offset += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(ProxyError::InvalidRequest(
                "Invalid protobuf varint".to_string(),
            ));
        }
    }

    Err(ProxyError::InvalidRequest(
        "Truncated protobuf varint".to_string(),
    ))
}

fn get_field_bytes(fields: &[ProtoField], field: u32) -> Option<&[u8]> {
    fields.iter().find_map(|candidate| {
        if candidate.field == field {
            if let ProtoValue::Bytes(bytes) = &candidate.value {
                return Some(bytes.as_slice());
            }
        }
        None
    })
}

fn get_all_field_bytes(fields: &[ProtoField], field: u32) -> Vec<&[u8]> {
    fields
        .iter()
        .filter_map(|candidate| {
            if candidate.field == field {
                if let ProtoValue::Bytes(bytes) = &candidate.value {
                    return Some(bytes.as_slice());
                }
            }
            None
        })
        .collect()
}

fn get_field_string(fields: &[ProtoField], field: u32) -> Option<String> {
    get_field_bytes(fields, field).map(|bytes| String::from_utf8_lossy(bytes).to_string())
}

fn get_field_varint(fields: &[ProtoField], field: u32) -> Option<u64> {
    fields.iter().find_map(|candidate| {
        if candidate.field == field {
            if let ProtoValue::Varint(value) = candidate.value {
                return Some(value);
            }
        }
        None
    })
}

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);
        if value == 0 {
            break;
        }
    }
    bytes
}

fn field_tag(field_num: u32, wire_type: u8) -> Vec<u8> {
    encode_varint((u64::from(field_num) << 3) | u64::from(wire_type))
}

fn write_varint_field(field_num: u32, value: u64) -> Vec<u8> {
    let mut bytes = field_tag(field_num, 0);
    bytes.extend_from_slice(&encode_varint(value));
    bytes
}

fn write_bytes_field(field_num: u32, data: &[u8]) -> Vec<u8> {
    let mut bytes = field_tag(field_num, 2);
    bytes.extend_from_slice(&encode_varint(data.len() as u64));
    bytes.extend_from_slice(data);
    bytes
}

fn write_string_field(field_num: u32, value: &str) -> Vec<u8> {
    write_bytes_field(field_num, value.as_bytes())
}

fn write_message_field(field_num: u32, value: Vec<u8>) -> Vec<u8> {
    write_bytes_field(field_num, &value)
}

fn write_fixed64_field(field_num: u32, value: [u8; 8]) -> Vec<u8> {
    let mut bytes = field_tag(field_num, 1);
    bytes.extend_from_slice(&value);
    bytes
}

fn write_fixed32_field(field_num: u32, value: [u8; 4]) -> Vec<u8> {
    let mut bytes = field_tag(field_num, 5);
    bytes.extend_from_slice(&value);
    bytes
}

fn build_timestamp() -> Vec<u8> {
    let now = chrono::Utc::now();
    let mut bytes = Vec::new();
    bytes.extend(write_varint_field(1, now.timestamp().max(0) as u64));
    bytes.extend(write_varint_field(2, now.timestamp_subsec_nanos() as u64));
    bytes
}

fn build_text_delta(message_id: &str, text: &str, token_count: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    if !text.is_empty() {
        bytes.extend(write_string_field(3, text));
    }
    if token_count > 0 {
        bytes.extend(write_varint_field(4, token_count));
    }
    bytes
}

fn build_thinking_delta(message_id: &str, text: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    bytes.extend(write_string_field(9, text));
    bytes
}

fn build_signature_delta(message_id: &str, signature: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    bytes.extend(write_string_field(10, signature));
    bytes
}

fn build_tool_call_delta(message_id: &str, tool_calls: &[ToolCallDelta]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    for call in tool_calls {
        let mut call_msg = Vec::new();
        call_msg.extend(write_string_field(1, &call.id));
        call_msg.extend(write_string_field(2, &call.name));
        call_msg.extend(write_string_field(3, &call.arguments_json));
        bytes.extend(write_message_field(6, call_msg));
    }
    bytes
}

fn build_stop_chunk(
    message_id: &str,
    stop_reason: u64,
    model_uid: &str,
    latency_ms: f64,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    bytes.extend(write_varint_field(5, stop_reason));
    bytes.extend(write_fixed64_field(12, latency_ms.to_le_bytes()));
    if !model_uid.is_empty() {
        bytes.extend(write_string_field(20, model_uid));
    }
    bytes
}

fn build_error_chunk(message_id: &str, error_text: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(write_string_field(1, message_id));
    bytes.extend(write_message_field(2, build_timestamp()));
    bytes.extend(write_string_field(3, error_text));
    bytes.extend(write_varint_field(5, STOP_ERROR));
    bytes
}

fn wrap_envelope(proto: &[u8]) -> Result<Vec<u8>, ProxyError> {
    let compressed = gzip_bytes(proto).map_err(|e| {
        ProxyError::Internal(format!("Failed to gzip Windsurf Connect envelope: {e}"))
    })?;
    let mut out = Vec::with_capacity(5 + compressed.len());
    out.push(1);
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

fn end_of_stream_envelope() -> Result<Vec<u8>, ProxyError> {
    let compressed = gzip_bytes(b"{}").map_err(|e| {
        ProxyError::Internal(format!("Failed to gzip Windsurf Connect trailers: {e}"))
    })?;
    let mut out = Vec::with_capacity(5 + compressed.len());
    out.push(3);
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

fn wrap_envelope_io(proto: &[u8]) -> Result<Bytes, std::io::Error> {
    wrap_envelope(proto)
        .map(Bytes::from)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

fn end_of_stream_envelope_io() -> Result<Bytes, std::io::Error> {
    end_of_stream_envelope()
        .map(Bytes::from)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

fn gzip_bytes(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes)?;
    encoder.finish()
}

fn gunzip_bytes(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(bytes);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

#[derive(Debug, Clone)]
struct SseEvent {
    event: String,
    data: Option<Value>,
    done: bool,
}

fn parse_sse_block(block: &str) -> Option<SseEvent> {
    let mut event_name = String::new();
    let mut data_lines = Vec::new();

    for raw_line in block.lines() {
        let mut line = raw_line.trim_start();
        if line.starts_with("data: event:") || line.starts_with("data: data:") {
            line = line.strip_prefix("data:").unwrap_or(line).trim_start();
        }
        if let Some(inline) = line.strip_prefix("event:") {
            if let Some((evt, data)) = inline.trim().split_once(" data:") {
                event_name = evt.trim().to_string();
                data_lines.push(data.trim_start().to_string());
                continue;
            }
        }

        if let Some(value) = strip_sse_field(line, "event") {
            event_name = value.trim().to_string();
        } else if let Some(value) = strip_sse_field(line, "data") {
            data_lines.push(value.to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    let data_text = data_lines.join("\n");
    if data_text.trim() == "[DONE]" {
        return Some(SseEvent {
            event: "done".to_string(),
            data: None,
            done: true,
        });
    }

    let data = serde_json::from_str::<Value>(&data_text).ok();
    if event_name.is_empty() {
        event_name = data
            .as_ref()
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            .or_else(|| {
                data.as_ref()
                    .and_then(|value| value.get("error"))
                    .map(|_| "error")
            })
            .unwrap_or_default()
            .to_string();
    }

    if event_name.is_empty() && data.is_none() {
        return None;
    }

    Some(SseEvent {
        event: event_name,
        data,
        done: false,
    })
}

#[derive(Debug, Clone, Default)]
struct ToolCallDelta {
    id: String,
    name: String,
    arguments_json: String,
}

enum StreamProcessor {
    Anthropic(AnthropicProcessor),
    OpenAI(OpenAIProcessor),
}

impl StreamProcessor {
    fn new(wire_api: UpstreamWireApi, message_id: String, model_uid: String) -> Self {
        match wire_api {
            UpstreamWireApi::AnthropicMessages => {
                Self::Anthropic(AnthropicProcessor::new(message_id, model_uid))
            }
            UpstreamWireApi::OpenAIResponses | UpstreamWireApi::OpenAIChatCompletions => {
                Self::OpenAI(OpenAIProcessor::new(message_id, model_uid))
            }
        }
    }

    fn message_id(&self) -> &str {
        match self {
            Self::Anthropic(processor) => &processor.message_id,
            Self::OpenAI(processor) => &processor.message_id,
        }
    }

    fn is_done(&self) -> bool {
        match self {
            Self::Anthropic(processor) => processor.done,
            Self::OpenAI(processor) => processor.done,
        }
    }

    fn model_uid(&self) -> &str {
        match self {
            Self::Anthropic(processor) => &processor.model_uid,
            Self::OpenAI(processor) => &processor.model_uid,
        }
    }

    fn process_sse_block(&mut self, block: &str, latency_ms: f64) -> Vec<Vec<u8>> {
        let Some(event) = parse_sse_block(block) else {
            return Vec::new();
        };

        if let Some(error) = error_message_from_sse(&event) {
            log::error!(
                "[Devin/Windsurf] Upstream SSE error event: message_id={}, model_uid={}, event={}, error={}",
                self.message_id(),
                self.model_uid(),
                event.event,
                super::sensitive_redaction::redact_sensitive_text(&error)
            );
            let frames = vec![build_error_chunk(
                self.message_id(),
                &format!("[API Error] {error}"),
            )];
            match self {
                Self::Anthropic(processor) => processor.done = true,
                Self::OpenAI(processor) => processor.done = true,
            }
            return frames;
        }

        match self {
            Self::Anthropic(processor) => processor.process(event, latency_ms),
            Self::OpenAI(processor) => processor.process(event, latency_ms),
        }
    }

    fn force_done(&mut self, latency_ms: f64) -> Vec<Vec<u8>> {
        match self {
            Self::Anthropic(processor) => processor.done(latency_ms),
            Self::OpenAI(processor) => processor.done(latency_ms),
        }
    }
}

fn error_message_from_sse(event: &SseEvent) -> Option<String> {
    if event.event.eq_ignore_ascii_case("error") {
        return event
            .data
            .as_ref()
            .and_then(error_value_message)
            .or_else(|| Some("upstream error event in SSE stream".to_string()));
    }

    event
        .data
        .as_ref()
        .and_then(|data| data.get("error"))
        .filter(|error| !error.is_null())
        .and_then(error_value_message)
}

fn error_value_message(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str().filter(|text| !text.trim().is_empty()) {
        return Some(text.to_string());
    }
    value
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/error/message").and_then(Value::as_str))
        .or_else(|| value.get("msg").and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .map(ToString::to_string)
}

struct AnthropicProcessor {
    message_id: String,
    model_uid: String,
    token_count: u64,
    done: bool,
    stop_reason: Option<String>,
    current_block_type: Option<String>,
    tool_id: String,
    tool_name: String,
    tool_args: String,
    signature: String,
}

impl AnthropicProcessor {
    fn new(message_id: String, model_uid: String) -> Self {
        Self {
            message_id,
            model_uid,
            token_count: 0,
            done: false,
            stop_reason: None,
            current_block_type: None,
            tool_id: String::new(),
            tool_name: String::new(),
            tool_args: String::new(),
            signature: String::new(),
        }
    }

    fn process(&mut self, event: SseEvent, latency_ms: f64) -> Vec<Vec<u8>> {
        if event.done {
            return self.done(latency_ms);
        }
        let mut frames = Vec::new();
        let data = event.data.unwrap_or(Value::Null);

        match event.event.as_str() {
            "content_block_start" => {
                let block = data.get("content_block").unwrap_or(&Value::Null);
                self.current_block_type = block
                    .get("type")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if self.current_block_type.as_deref() == Some("tool_use") {
                    self.tool_id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.tool_name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.tool_args.clear();
                } else if self.current_block_type.as_deref() == Some("thinking") {
                    self.signature.clear();
                }
            }
            "content_block_delta" => {
                let delta = data.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            self.token_count += 1;
                            frames.push(build_text_delta(&self.message_id, text, self.token_count));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                            frames.push(build_thinking_delta(&self.message_id, text));
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                            self.tool_args.push_str(partial);
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) = delta.get("signature").and_then(Value::as_str) {
                            self.signature.push_str(signature);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => self.flush_current_block(&mut frames),
            "message_delta" => {
                if let Some(reason) = data.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_string());
                }
            }
            "message_stop" | "done" => frames.extend(self.done(latency_ms)),
            _ => {}
        }

        frames
    }

    fn flush_current_block(&mut self, frames: &mut Vec<Vec<u8>>) {
        match self.current_block_type.as_deref() {
            Some("tool_use") => {
                frames.push(build_tool_call_delta(
                    &self.message_id,
                    &[ToolCallDelta {
                        id: self.tool_id.clone(),
                        name: self.tool_name.clone(),
                        arguments_json: self.tool_args.clone(),
                    }],
                ));
                self.tool_id.clear();
                self.tool_name.clear();
                self.tool_args.clear();
            }
            Some("thinking") if !self.signature.is_empty() => {
                frames.push(build_signature_delta(&self.message_id, &self.signature));
                self.signature.clear();
            }
            _ => {}
        }
        self.current_block_type = None;
    }

    fn done(&mut self, latency_ms: f64) -> Vec<Vec<u8>> {
        if self.done {
            return Vec::new();
        }
        let mut frames = Vec::new();
        if self.current_block_type.is_some() {
            self.flush_current_block(&mut frames);
        }
        let stop = match self.stop_reason.as_deref() {
            Some("tool_use") => STOP_FUNCTION_CALL,
            Some("max_tokens") => STOP_MAX_TOKENS,
            _ => STOP_PATTERN,
        };
        frames.push(build_stop_chunk(
            &self.message_id,
            stop,
            &self.model_uid,
            latency_ms,
        ));
        self.done = true;
        frames
    }
}

struct OpenAIProcessor {
    message_id: String,
    model_uid: String,
    token_count: u64,
    text_frame_count: u64,
    thinking_frame_count: u64,
    tool_frame_count: u64,
    done: bool,
    stop_reason: Option<String>,
    tool_calls: BTreeMap<usize, ToolCallDelta>,
    item_types: BTreeMap<usize, String>,
    item_phases: BTreeMap<usize, String>,
}

impl OpenAIProcessor {
    fn new(message_id: String, model_uid: String) -> Self {
        Self {
            message_id,
            model_uid,
            token_count: 0,
            text_frame_count: 0,
            thinking_frame_count: 0,
            tool_frame_count: 0,
            done: false,
            stop_reason: None,
            tool_calls: BTreeMap::new(),
            item_types: BTreeMap::new(),
            item_phases: BTreeMap::new(),
        }
    }

    fn process(&mut self, event: SseEvent, latency_ms: f64) -> Vec<Vec<u8>> {
        let events = normalize_openai_event(event);
        let mut frames = Vec::new();

        for event in events {
            if event.done {
                frames.extend(self.done(latency_ms));
                continue;
            }

            let data = event.data.unwrap_or(Value::Null);
            match event.event.as_str() {
                "response.reasoning.delta" | "response.reasoning_summary_text.delta" => {
                    if let Some(delta) = data.get("delta").and_then(Value::as_str) {
                        self.thinking_frame_count += 1;
                        frames.push(build_thinking_delta(&self.message_id, delta));
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = data.get("delta").and_then(Value::as_str) {
                        let index = data
                            .get("output_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize;
                        let is_thinking = self
                            .item_types
                            .get(&index)
                            .is_some_and(|kind| kind == "reasoning")
                            || self
                                .item_phases
                                .get(&index)
                                .is_some_and(|phase| phase == "thinking");
                        if is_thinking {
                            self.thinking_frame_count += 1;
                            frames.push(build_thinking_delta(&self.message_id, delta));
                        } else {
                            self.token_count += 1;
                            self.text_frame_count += 1;
                            frames.push(build_text_delta(
                                &self.message_id,
                                delta,
                                self.token_count,
                            ));
                        }
                    }
                }
                "response.output_item.added" => {
                    let index = data
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let item = data.get("item").unwrap_or(&Value::Null);
                    self.record_openai_output_item(index, item, false);
                }
                "response.output_item.done" => {
                    let index = data
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let item = data.get("item").unwrap_or(&Value::Null);
                    self.record_openai_output_item(index, item, true);
                }
                "response.function_call_arguments.delta" => {
                    let index = data
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    if let Some(call) = self.tool_calls.get_mut(&index) {
                        if let Some(delta) = data.get("delta").and_then(Value::as_str) {
                            call.arguments_json.push_str(delta);
                        }
                    }
                }
                "response.function_call_arguments.done" => {
                    let index = data
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    if let Some(call) = self.tool_calls.get_mut(&index) {
                        if let Some(arguments) = data.get("arguments").and_then(Value::as_str) {
                            call.arguments_json = arguments.to_string();
                        }
                    }
                }
                "response.completed" => {
                    let response = data.get("response").unwrap_or(&data);
                    if response.get("status").and_then(Value::as_str) == Some("completed") {
                        let mut has_tool_calls = false;
                        if let Some(items) = response.get("output").and_then(Value::as_array) {
                            for (index, item) in items.iter().enumerate() {
                                if item.get("type").and_then(Value::as_str) == Some("function_call")
                                {
                                    self.record_openai_output_item(index, item, true);
                                    if self.tool_calls.get(&index).is_some_and(|call| {
                                        !call.name.is_empty() || !call.arguments_json.is_empty()
                                    }) {
                                        has_tool_calls = true;
                                    }
                                }
                            }
                        }
                        if !has_tool_calls {
                            has_tool_calls = self.tool_calls.values().any(|call| {
                                !call.name.is_empty() || !call.arguments_json.is_empty()
                            });
                        }
                        self.stop_reason = Some(if has_tool_calls {
                            "tool_calls".to_string()
                        } else {
                            "stop".to_string()
                        });
                    }
                    frames.extend(self.done(latency_ms));
                }
                "response.failed" => {
                    let message = data
                        .pointer("/response/error/message")
                        .and_then(Value::as_str)
                        .unwrap_or("response.failed event received");
                    frames.push(build_error_chunk(&self.message_id, message));
                    self.done = true;
                }
                _ => {}
            }
        }

        frames
    }

    fn record_openai_output_item(&mut self, index: usize, item: &Value, complete: bool) {
        if let Some(kind) = item.get("type").and_then(Value::as_str) {
            self.item_types.insert(index, kind.to_string());
        }
        if let Some(phase) = item.get("phase").and_then(Value::as_str) {
            self.item_phases.insert(index, phase.to_string());
        }
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return;
        }

        let call = self
            .tool_calls
            .entry(index)
            .or_insert_with(|| ToolCallDelta {
                id: String::new(),
                name: String::new(),
                arguments_json: String::new(),
            });

        if let Some(id) = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            call.id = id.to_string();
        }
        if let Some(name) = item
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            call.name = name.to_string();
        }
        if complete {
            if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                call.arguments_json = arguments.to_string();
            }
        } else if call.arguments_json.is_empty() {
            if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                call.arguments_json = arguments.to_string();
            }
        }
    }

    fn done(&mut self, latency_ms: f64) -> Vec<Vec<u8>> {
        if self.done {
            return Vec::new();
        }
        let mut frames = Vec::new();
        if !self.tool_calls.is_empty() {
            let calls = self
                .tool_calls
                .values()
                .filter(|call| !call.name.is_empty() || !call.arguments_json.is_empty())
                .cloned()
                .collect::<Vec<_>>();
            if !calls.is_empty() {
                self.tool_frame_count += calls.len() as u64;
                frames.push(build_tool_call_delta(&self.message_id, &calls));
            }
        }
        if self.stop_reason.is_none() {
            self.stop_reason = Some(if self.tool_frame_count == 0 {
                "stop".to_string()
            } else {
                "tool_calls".to_string()
            });
        }
        if self.text_frame_count == 0 && self.tool_frame_count == 0 {
            log::warn!(
                "[Devin/Windsurf] OpenAI stream completed without visible text/tool frames: message_id={}, thinking_frames={}, stop_reason={:?}",
                self.message_id,
                self.thinking_frame_count,
                self.stop_reason
            );
        }
        let stop = match self.stop_reason.as_deref() {
            Some("tool_calls") => STOP_FUNCTION_CALL,
            Some("length") => STOP_MAX_TOKENS,
            _ => STOP_PATTERN,
        };
        frames.push(build_stop_chunk(
            &self.message_id,
            stop,
            &self.model_uid,
            latency_ms,
        ));
        self.done = true;
        frames
    }
}

fn normalize_openai_event(event: SseEvent) -> Vec<SseEvent> {
    if event.done {
        return vec![event];
    }
    let Some(data) = event.data.as_ref() else {
        return vec![event];
    };
    if data.get("type").and_then(Value::as_str).is_some() {
        return vec![event];
    }

    let Some(choice) = data
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return vec![event];
    };

    let mut events = Vec::new();
    let delta = choice.get("delta").unwrap_or(&Value::Null);

    if let Some(reasoning) = delta
        .get("reasoning")
        .or_else(|| delta.get("reasoning_content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        events.push(SseEvent {
            event: "response.reasoning.delta".to_string(),
            data: Some(json!({ "delta": reasoning, "output_index": 0 })),
            done: false,
        });
    }

    if let Some(text) = delta
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        events.push(SseEvent {
            event: "response.output_text.delta".to_string(),
            data: Some(json!({ "delta": text, "output_index": 0 })),
            done: false,
        });
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            if tool_call.get("id").and_then(Value::as_str).is_some()
                || function.get("name").and_then(Value::as_str).is_some()
            {
                events.push(SseEvent {
                    event: "response.output_item.added".to_string(),
                    data: Some(json!({
                        "output_index": index,
                        "item": {
                            "type": "function_call",
                            "call_id": tool_call.get("id").and_then(Value::as_str).unwrap_or_default(),
                            "name": function.get("name").and_then(Value::as_str).unwrap_or_default()
                        }
                    })),
                    done: false,
                });
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                events.push(SseEvent {
                    event: "response.function_call_arguments.delta".to_string(),
                    data: Some(json!({ "output_index": index, "delta": arguments })),
                    done: false,
                });
            }
        }
    }

    if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
        events.push(SseEvent {
            event: "response.completed".to_string(),
            data: Some(json!({
                "response": {
                    "status": "completed",
                    "finish_reason": finish_reason,
                    "output": [],
                    "usage": data.get("usage").cloned().unwrap_or_else(|| json!({}))
                }
            })),
            done: false,
        });
    }

    if events.is_empty() {
        vec![event]
    } else {
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::providers::{transform, transform_responses};

    fn minimal_get_chat_request_proto() -> Vec<u8> {
        let mut message = Vec::new();
        message.extend(write_string_field(1, "msg_1"));
        message.extend(write_varint_field(2, SOURCE_USER));
        message.extend(write_string_field(3, "ping"));

        let mut root = Vec::new();
        root.extend(write_string_field(2, "system prompt"));
        root.extend(write_message_field(3, message));
        root.extend(write_string_field(21, "gpt-5.5"));
        root
    }

    fn image_proto(data: &str, mime_type: &str) -> Vec<u8> {
        let mut image = Vec::new();
        image.extend(write_string_field(1, data));
        image.extend(write_string_field(2, mime_type));
        image
    }

    fn tool_call_proto(id: &str, name: &str, arguments_json: &str) -> Vec<u8> {
        let mut tool_call = Vec::new();
        tool_call.extend(write_string_field(1, id));
        tool_call.extend(write_string_field(2, name));
        tool_call.extend(write_string_field(3, arguments_json));
        tool_call
    }

    fn tool_definition_proto(name: &str, description: &str, schema: &str) -> Vec<u8> {
        let mut tool = Vec::new();
        tool.extend(write_string_field(1, name));
        tool.extend(write_string_field(2, description));
        tool.extend(write_string_field(3, schema));
        tool
    }

    fn tool_choice_any_proto() -> Vec<u8> {
        let mut choice = Vec::new();
        choice.extend(write_string_field(1, "any"));
        choice
    }

    #[test]
    fn upstream_wire_api_detects_provider_prefixed_endpoints() {
        assert_eq!(
            UpstreamWireApi::from_endpoint("/api/saas/openai/v1/responses"),
            UpstreamWireApi::OpenAIResponses
        );
        assert_eq!(
            UpstreamWireApi::from_endpoint("/api/saas/openai/v1/responses?stream=true"),
            UpstreamWireApi::OpenAIResponses
        );
        assert_eq!(
            UpstreamWireApi::from_endpoint("/api/saas/anthropic/v1/messages"),
            UpstreamWireApi::AnthropicMessages
        );
        assert_eq!(
            UpstreamWireApi::from_endpoint("/api/saas/openai/v1/chat/completions"),
            UpstreamWireApi::OpenAIChatCompletions
        );
    }

    #[test]
    fn connect_envelope_round_trips_gzip_payload() {
        let proto = minimal_get_chat_request_proto();
        let envelope = wrap_envelope(&proto).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("application/connect+proto"),
        );
        let unwrapped = unwrap_connect_request(&envelope, &headers).unwrap();
        assert_eq!(unwrapped, proto);
    }

    #[test]
    fn parses_minimal_get_chat_message_request() {
        let proto = minimal_get_chat_request_proto();
        let parsed = parse_get_chat_message_request(&proto, &HeaderMap::new()).unwrap();
        assert_eq!(parsed.system_prompt, "system prompt");
        assert_eq!(parsed.requested_model.as_deref(), Some("gpt-5.5"));
        assert_eq!(parsed.initiator, "user");
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.messages[0]["role"], "user");
        assert_eq!(parsed.messages[0]["content"], "ping");
    }

    #[test]
    fn rich_windsurf_request_preserves_canonical_and_degrades_provider_only_fields() {
        let mut system_prompt_message = Vec::new();
        system_prompt_message.extend(write_varint_field(2, SOURCE_SYSTEM_PROMPT));
        system_prompt_message.extend(write_string_field(3, "workspace rules"));

        let mut user_message = Vec::new();
        user_message.extend(write_string_field(1, "msg_user"));
        user_message.extend(write_varint_field(2, SOURCE_USER));
        user_message.extend(write_string_field(3, "look at this"));
        user_message.extend(write_message_field(10, image_proto("aW1n", "image/png")));

        let mut assistant_message = Vec::new();
        assistant_message.extend(write_string_field(1, "msg_assistant"));
        assistant_message.extend(write_varint_field(2, SOURCE_SYSTEM));
        assistant_message.extend(write_string_field(3, "I will call a tool"));
        assistant_message.extend(write_message_field(
            6,
            tool_call_proto("call_1", "Read", r#"{"path":"src/main.rs"}"#),
        ));
        assistant_message.extend(write_string_field(11, "private thinking"));
        assistant_message.extend(write_string_field(12, "sig_123"));

        let mut tool_result_message = Vec::new();
        tool_result_message.extend(write_string_field(1, "msg_tool"));
        tool_result_message.extend(write_varint_field(2, SOURCE_TOOL));
        tool_result_message.extend(write_string_field(3, "file content"));
        tool_result_message.extend(write_string_field(7, "call_1"));
        tool_result_message.extend(write_varint_field(9, 1));

        let mut root = Vec::new();
        root.extend(write_string_field(2, "base system"));
        root.extend(write_message_field(3, system_prompt_message));
        root.extend(write_message_field(3, user_message));
        root.extend(write_message_field(3, assistant_message));
        root.extend(write_message_field(3, tool_result_message));
        root.extend(write_message_field(
            10,
            tool_definition_proto(
                "Read",
                "Read a file",
                r#"{"type":"object","properties":{"path":{"type":"string"}}}"#,
            ),
        ));
        root.extend(write_message_field(12, tool_choice_any_proto()));
        root.extend(write_string_field(21, "MODEL_PRIVATE_11"));

        let parsed = parse_get_chat_message_request(&root, &HeaderMap::new()).unwrap();
        assert_eq!(parsed.system_prompt, "base system\nworkspace rules");
        assert_eq!(parsed.requested_model.as_deref(), Some("MODEL_PRIVATE_11"));
        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[0]["content"][0]["type"], "image");
        assert_eq!(parsed.messages[1]["content"][0]["type"], "thinking");
        assert_eq!(parsed.messages[1]["content"][0]["signature"], "sig_123");
        assert_eq!(parsed.messages[1]["content"][2]["type"], "tool_use");
        assert_eq!(parsed.messages[2]["content"][0]["is_error"], true);
        assert_eq!(parsed.tools[0]["name"], "Read");
        assert_eq!(parsed.tool_choice.as_ref().unwrap()["type"], "any");

        let canonical_body = parsed.to_anthropic_body();
        assert_eq!(
            canonical_body["_cc_switch_canonical_api"],
            "anthropic_messages"
        );
        assert_eq!(
            canonical_body["messages"][1]["content"][0]["type"],
            "thinking"
        );
        assert_eq!(
            canonical_body["messages"][1]["content"][0]["signature"],
            "sig_123"
        );
        assert_eq!(
            canonical_body["messages"][2]["content"][0]["is_error"],
            true
        );

        let responses_body =
            transform_responses::anthropic_to_responses(canonical_body.clone(), None, false, false)
                .unwrap();
        assert_eq!(responses_body["model"], "MODEL_PRIVATE_11");
        assert_eq!(
            responses_body["instructions"],
            "base system\nworkspace rules"
        );
        assert_eq!(
            responses_body["input"][0]["content"][0]["image_url"],
            "data:image/png;base64,aW1n"
        );
        assert_eq!(
            responses_body["input"][1]["content"][0]["type"],
            "output_text"
        );
        assert_eq!(responses_body["input"][2]["type"], "function_call");
        assert_eq!(responses_body["input"][3]["type"], "function_call_output");
        assert_eq!(responses_body["tool_choice"], "required");
        // Thinking is dropped during responses conversion — no cc-switch tags.
        assert!(!responses_body["input"]
            .to_string()
            .contains("cc-switch:thinking"));
        assert!(responses_body["input"]
            .to_string()
            .contains("[cc-switch:tool-result-error]"));

        let chat_body = transform::anthropic_to_openai(canonical_body).unwrap();
        assert_eq!(chat_body["model"], "MODEL_PRIVATE_11");
        assert_eq!(chat_body["messages"][0]["role"], "system");
        assert_eq!(
            chat_body["messages"][1]["content"][0]["image_url"]["url"],
            "data:image/png;base64,aW1n"
        );
        assert_eq!(
            chat_body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            r#"{"path":"src/main.rs"}"#
        );
        assert!(chat_body["messages"][2].get("reasoning_content").is_none());
        // Thinking is not leaked into visible content in chat conversion.
        assert!(!chat_body["messages"][2]["content"]
            .to_string()
            .contains("cc-switch:thinking"));
        assert_eq!(chat_body["messages"][3]["role"], "tool");
        assert!(chat_body["messages"][3].get("is_error").is_none());
        assert!(chat_body["messages"][3]["content"]
            .as_str()
            .unwrap()
            .contains(r#"<cc-switch:tool_result is_error="true">"#));
        assert_eq!(chat_body["tool_choice"], "required");

        let round_tripped = transform::openai_chat_request_to_anthropic(chat_body).unwrap();
        assert_eq!(
            round_tripped["system"][0]["text"],
            "base system\nworkspace rules"
        );
        assert_eq!(
            round_tripped["system"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(round_tripped["messages"][0]["content"][0]["type"], "image");
        assert_eq!(round_tripped["messages"][1]["content"][0]["type"], "text");
        // Thinking dropped → tool_use is now at index 1 (was index 2)
        assert_eq!(
            round_tripped["messages"][1]["content"][1]["type"],
            "tool_use"
        );
        assert!(round_tripped["messages"][1]["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block.get("type").and_then(Value::as_str) != Some("thinking")));
        // Unsigned thinking is now dropped, not wrapped in visible tags.
        assert_eq!(round_tripped["messages"][1]["content"][0]["type"], "text");
        assert!(!round_tripped["messages"][1]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("cc-switch:thinking"));
        assert!(round_tripped["messages"][2]["content"][0]
            .get("is_error")
            .is_none());
        assert!(round_tripped["messages"][2]["content"][0]["content"]
            .as_str()
            .unwrap()
            .contains(r#"<cc-switch:tool_result is_error="true">"#));
    }

    #[test]
    fn unsigned_historical_thinking_degrades_to_text_for_anthropic_replay() {
        let mut assistant_message = Vec::new();
        assistant_message.extend(write_varint_field(2, SOURCE_SYSTEM));
        assistant_message.extend(write_string_field(3, "I will inspect the files"));
        assistant_message.extend(write_string_field(11, "Need to inspect before editing."));
        assistant_message.extend(write_message_field(
            6,
            tool_call_proto("call_1", "Read", r#"{"path":"src/main.rs"}"#),
        ));

        let mut root = Vec::new();
        root.extend(write_message_field(3, assistant_message));

        let parsed = parse_get_chat_message_request(&root, &HeaderMap::new()).unwrap();
        assert_eq!(parsed.messages.len(), 1);
        let content = parsed.messages[0]["content"].as_array().unwrap();
        // Unsigned thinking is dropped — no thinking blocks, no cc-switch tags.
        assert!(content
            .iter()
            .all(|block| block.get("type").and_then(Value::as_str) != Some("thinking")));
        assert!(content.iter().all(|block| {
            block
                .get("text")
                .and_then(Value::as_str)
                .map(|t| !t.contains("cc-switch:thinking"))
                .unwrap_or(true)
        }));
        // Prompt text and tool_use should still be present.
        assert!(content
            .iter()
            .any(|block| block.get("text").and_then(Value::as_str)
                == Some("I will inspect the files")));
        assert!(content
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use")));
    }

    #[test]
    fn chat_sse_delta_builds_windsurf_text_and_stop_frames() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::OpenAIChatCompletions,
            "m1".to_string(),
            "gpt-5.5".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}"#,
            1.0,
        );
        assert_eq!(frames.len(), 1);
        let fields = parse_fields(&frames[0]).unwrap();
        assert_eq!(get_field_string(&fields, 3).as_deref(), Some("pong"));

        let frames = processor.process_sse_block(
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            2.0,
        );
        assert_eq!(frames.len(), 1);
        let fields = parse_fields(&frames[0]).unwrap();
        assert_eq!(get_field_varint(&fields, 5), Some(STOP_PATTERN));
    }

    #[test]
    fn chat_sse_tool_finish_without_tool_delta_does_not_emit_empty_tool_call() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::OpenAIChatCompletions,
            "m_tool_finish".to_string(),
            "gpt-5.5".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            1.0,
        );

        assert_eq!(frames.len(), 1);
        let fields = parse_fields(&frames[0]).unwrap();
        assert!(get_field_bytes(&fields, 6).is_none());
        assert_eq!(get_field_varint(&fields, 5), Some(STOP_PATTERN));
    }

    #[test]
    fn responses_sse_maps_reasoning_tool_call_and_stop_frames() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::OpenAIResponses,
            "m2".to_string(),
            "gpt-5.5".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.reasoning.delta","delta":"thinking"}"#,
            1.0,
        );
        assert_eq!(frames.len(), 1);
        let fields = parse_fields(&frames[0]).unwrap();
        assert_eq!(get_field_string(&fields, 9).as_deref(), Some("thinking"));

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"Read"}}"#,
            2.0,
        );
        assert!(frames.is_empty());

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":\"src/main.rs\"}"}"#,
            3.0,
        );
        assert!(frames.is_empty());

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call"}]}}"#,
            4.0,
        );
        assert_eq!(frames.len(), 2);
        let tool_fields = parse_fields(&frames[0]).unwrap();
        let call = get_field_bytes(&tool_fields, 6).unwrap();
        let call_fields = parse_fields(call).unwrap();
        assert_eq!(get_field_string(&call_fields, 1).as_deref(), Some("call_1"));
        assert_eq!(get_field_string(&call_fields, 2).as_deref(), Some("Read"));
        assert_eq!(
            get_field_string(&call_fields, 3).as_deref(),
            Some(r#"{"path":"src/main.rs"}"#)
        );

        let stop_fields = parse_fields(&frames[1]).unwrap();
        assert_eq!(get_field_varint(&stop_fields, 5), Some(STOP_FUNCTION_CALL));
    }

    #[test]
    fn responses_sse_maps_completed_function_call_item() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::OpenAIResponses,
            "m_done".to_string(),
            "gpt-5.3-codex".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_done","name":"Edit","arguments":"{\"path\":\"src/lib.rs\",\"old\":\"a\",\"new\":\"b\"}"}}"#,
            1.0,
        );
        assert!(frames.is_empty());

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"call_done","name":"Edit","arguments":"{\"path\":\"src/lib.rs\",\"old\":\"a\",\"new\":\"b\"}"}]}}"#,
            2.0,
        );
        assert_eq!(frames.len(), 2);
        let tool_fields = parse_fields(&frames[0]).unwrap();
        let call = get_field_bytes(&tool_fields, 6).unwrap();
        let call_fields = parse_fields(call).unwrap();
        assert_eq!(
            get_field_string(&call_fields, 1).as_deref(),
            Some("call_done")
        );
        assert_eq!(get_field_string(&call_fields, 2).as_deref(), Some("Edit"));
        assert_eq!(
            get_field_string(&call_fields, 3).as_deref(),
            Some(r#"{"path":"src/lib.rs","old":"a","new":"b"}"#)
        );
        let stop_fields = parse_fields(&frames[1]).unwrap();
        assert_eq!(get_field_varint(&stop_fields, 5), Some(STOP_FUNCTION_CALL));
    }

    #[test]
    fn responses_sse_maps_function_call_arguments_done() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::OpenAIResponses,
            "m_args_done".to_string(),
            "gpt-5.3-codex".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_args","name":"Read"}}"#,
            1.0,
        );
        assert!(frames.is_empty());

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"path\":\"Cargo.toml\"}"}"#,
            2.0,
        );
        assert!(frames.is_empty());

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call"}]}}"#,
            3.0,
        );
        assert_eq!(frames.len(), 2);
        let tool_fields = parse_fields(&frames[0]).unwrap();
        let call = get_field_bytes(&tool_fields, 6).unwrap();
        let call_fields = parse_fields(call).unwrap();
        assert_eq!(
            get_field_string(&call_fields, 3).as_deref(),
            Some(r#"{"path":"Cargo.toml"}"#)
        );
    }

    #[test]
    fn prefixed_responses_endpoint_maps_text_delta_to_windsurf_text_frame() {
        let mut processor = StreamProcessor::new(
            UpstreamWireApi::from_endpoint("/api/saas/openai/v1/responses"),
            "m3".to_string(),
            "GPT 5.3-codex".to_string(),
        );

        let frames = processor.process_sse_block(
            r#"data: {"type":"response.output_text.delta","content_index":0,"delta":"你好","output_index":0}"#,
            1.0,
        );
        assert_eq!(frames.len(), 1);
        let fields = parse_fields(&frames[0]).unwrap();
        assert_eq!(get_field_string(&fields, 3).as_deref(), Some("你好"));
        assert_eq!(get_field_varint(&fields, 4), Some(1));
    }

    #[test]
    fn register_user_response_rewrites_api_server_url() {
        let mut proto = Vec::new();
        proto.extend(write_string_field(1, "api-key"));
        proto.extend(write_string_field(2, "user"));
        proto.extend(write_string_field(
            3,
            "https://server.self-serve.windsurf.com",
        ));

        let rewritten =
            rewrite_register_user_response_body(&proto, "http://127.0.0.1:15721/_route/api_server")
                .unwrap()
                .unwrap();

        let fields = parse_fields(&rewritten).unwrap();
        assert_eq!(
            get_field_string(&fields, 3).as_deref(),
            Some("http://127.0.0.1:15721/_route/api_server")
        );
    }

    #[test]
    fn register_user_response_rewrites_gzip_connect_envelope() {
        let mut proto = Vec::new();
        proto.extend(write_string_field(1, "api-key"));
        proto.extend(write_string_field(2, "user"));
        proto.extend(write_string_field(
            3,
            "https://server.self-serve.windsurf.com",
        ));
        let envelope = wrap_envelope(&proto).unwrap();

        let rewritten = rewrite_register_user_response_body(
            &envelope,
            "http://localhost:15721/_route/api_server",
        )
        .unwrap()
        .unwrap();

        assert_eq!(rewritten[0], 0);
        let len =
            u32::from_be_bytes([rewritten[1], rewritten[2], rewritten[3], rewritten[4]]) as usize;
        assert_eq!(len, rewritten.len() - 5);
        let fields = parse_fields(&rewritten[5..]).unwrap();
        assert_eq!(
            get_field_string(&fields, 3).as_deref(),
            Some("http://localhost:15721/_route/api_server")
        );
    }
}

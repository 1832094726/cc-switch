//! Anthropic Messages SSE → OpenAI Chat Completions SSE conversion.
//!
//! Converts upstream Anthropic SSE events into Chat Completions SSE chunks
//! so they can be fed into the existing Chat→Responses pipeline.

use crate::proxy::sse::{strip_nested_data_prefix, strip_sse_field, take_sse_block};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Convert an Anthropic Messages SSE byte stream into an OpenAI Chat
/// Completions SSE byte stream.
pub fn create_chat_sse_stream_from_anthropic<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut state = AnthropicToChatState::default();
        let mut diag_chunk_count: u64 = 0;
        let mut diag_total_bytes: u64 = 0;
        let mut diag_event_count: u64 = 0;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    diag_chunk_count += 1;
                    diag_total_bytes += bytes.len() as u64;
                    crate::proxy::sse::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(block) = take_sse_block(&mut buffer) {
                        if block.trim().is_empty() {
                            continue;
                        }

                        // 兼容标准单层 Anthropic SSE 以及被网关双层包裹的 SSE
                        // （每个物理行被再套一层 `data: `，包括内层的 `event:` 行）。
                        // handle_anthropic_event 依赖 JSON 的 `type` 字段分派，
                        // 因此逐行剥离嵌套 `data:` 前缀后，只需提取内层 JSON。
                        for raw_line in block.lines() {
                            let inner = strip_nested_data_prefix(raw_line);
                            let trimmed = inner.trim();
                            if trimmed.is_empty() || trimmed == "[DONE]" {
                                continue;
                            }
                            // 跳过内层的 `event:` 行（事件类型由 JSON `type` 决定）。
                            if strip_sse_field(trimmed, "event").is_some() {
                                continue;
                            }

                            let parsed: Value = match serde_json::from_str(trimmed) {
                                Ok(v) => v,
                                Err(e) => {
                                    log::warn!(
                                        "[AnthropicToChat] 无法解析 SSE data: {e}, data={}",
                                        truncate_for_log(trimmed)
                                    );
                                    continue;
                                }
                            };
                            diag_event_count += 1;
                            if log::log_enabled!(log::Level::Debug) {
                                log::debug!(
                                    "[AnthropicToChat] 解析事件 #{diag_event_count}: type={:?}, raw={}",
                                    parsed.get("type").and_then(|t| t.as_str()),
                                    truncate_for_log(trimmed)
                                );
                            }

                            for chat_chunk in state.handle_anthropic_event(&parsed) {
                                yield Ok(chat_chunk);
                            }

                            if state.done && !state.sent_done {
                                state.sent_done = true;
                                yield Ok(Bytes::from("data: [DONE]\n\n"));
                            }
                        }
                    }
                }
                Err(e) => {
                    // 上游传输层错误：记录并中断，交由外层 create_logged_passthrough_stream
                    // 生成错误事件。
                    log::warn!(
                        "[AnthropicToChat] 上游流传输错误: {e}, chunks={diag_chunk_count}, bytes={diag_total_bytes}, events={diag_event_count}"
                    );
                    state.upstream_error = Some(format!("Upstream stream transport error: {e}"));
                    break;
                }
            }
        }

        log::debug!(
            "[AnthropicToChat] 上游流结束: chunks={diag_chunk_count}, bytes={diag_total_bytes}, events={diag_event_count}, done={}, finish_reason={:?}, upstream_error={:?}",
            state.done,
            state.finish_reason,
            state.upstream_error
        );

        // Finalize: if upstream stream ended without message_stop, synthesize
        // finish_reason chunk + [DONE] so downstream Chat→Responses converter
        // does not see a truncated stream.
        if !state.done {
            if let Some(err) = state.upstream_error.take() {
                // 上游中途报错/断开：不能伪装成正常完成，向下游发送 Chat SSE error 事件。
                log::warn!("[AnthropicToChat] 上游异常中断，转发错误: {err}");
                let error_chunk = json!({
                    "error": {
                        "message": err,
                        "type": "upstream_error"
                    }
                });
                yield Ok(Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&error_chunk).unwrap_or_default()
                )));
                yield Ok(Bytes::from("data: [DONE]\n\n"));
            } else {
                // 上游正常关闭但缺 message_stop：合成 finish_reason + [DONE]。
                state.done = true;
                state.sent_done = true;
                let fr = state.finish_reason.as_deref().unwrap_or("stop");
                let mut chunk = json!({
                    "delta": {},
                    "finish_reason": fr
                });
                if let Some(u) = state.chat_usage() {
                    chunk["usage"] = u;
                }
                yield Ok(state.build_chat_chunk(chunk));
                yield Ok(Bytes::from("data: [DONE]\n\n"));
            }
        }
    }
}

fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 500;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}...(truncated {} bytes)", &s[..MAX], s.len() - MAX)
    }
}

#[derive(Debug, Default)]
struct AnthropicToChatState {
    message_id: String,
    model: String,
    created: u64,
    /// "chatcmpl-" prefix ID derived from Anthropic message id.
    chat_id: String,
    current_content_block_type: Option<String>,
    current_content_block_index: Option<u32>,
    finish_reason: Option<String>,
    done: bool,
    sent_done: bool,
    usage: Option<Value>,
    tool_use_states: HashMap<u32, ToolUseState>,
    /// Reasoning/thinking accumulated per tool-use index → append as
    /// reasoning_content in the next Chat delta.
    pending_reasoning: String,
    /// 上游中途发送的 error 事件或传输错误，用于向下游转发而非伪装为正常完成。
    upstream_error: Option<String>,
}

#[derive(Debug, Default)]
struct ToolUseState {
    id: String,
    name: String,
    arguments: String,
    started: bool,
    done: bool,
}

impl AnthropicToChatState {
    fn handle_anthropic_event(&mut self, event: &Value) -> Vec<Bytes> {
        let event_type = event.get("type").and_then(|t| t.as_str());
        let mut out: Vec<Bytes> = Vec::new();

        // 兜底：部分上游（如 JoyCode 内容审查）在流里直接返回
        // `{"error":{"code":400,"message":"sensitive contain:[...]"}}`，
        // 顶层没有 `type` 字段，导致 event_type=None 落入 `_ => {}` 被静默丢弃：
        // 流无 message_stop、finish_reason=None、upstream_error=None，最终静默结束，
        // 客户端收不到任何完成或错误信号而一直卡住。这里显式识别顶层 error 对象，
        // 记录为 upstream_error，交由 finalize 向下游转发标准错误事件。
        if event_type.is_none() {
            if let Some(err_obj) = event.get("error").filter(|e| e.is_object()) {
                let err_type = err_obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .or_else(|| err_obj.get("code").and_then(|v| v.as_str()))
                    .unwrap_or("error");
                let err_msg = err_obj
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("upstream returned error object without type");
                log::warn!(
                    "[AnthropicToChat] 上游错误对象(无 type): type={err_type}, message={err_msg}"
                );
                if self.upstream_error.is_none() {
                    self.upstream_error = Some(format!("Upstream error ({err_type}): {err_msg}"));
                }
                return out;
            }
        }

        match event_type {
            Some("message_start") => {
                if let Some(msg) = event.get("message") {
                    if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                        self.message_id = id.to_string();
                        self.chat_id = format!("chatcmpl-{}", id);
                    }
                    if let Some(model) = msg.get("model").and_then(|v| v.as_str()) {
                        self.model = model.to_string();
                    }
                    if let Some(usage) = msg.get("usage") {
                        self.usage = Some(usage.clone());
                    }
                }
                self.created = now_secs();
                // Emit first chat chunk with role
                out.push(self.build_chat_chunk(json!({"role": "assistant"})));
            }

            Some("content_block_start") => {
                let index = event
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .map(|i| i as u32);
                let block = event.get("content_block");
                let block_type = block
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                match block_type.as_str() {
                    "text" => {
                        self.current_content_block_index = index;
                        self.current_content_block_type = Some("text".into());
                        if !self.pending_reasoning.is_empty() {
                            let reasoning = std::mem::take(&mut self.pending_reasoning);
                            out.push(self.build_chat_chunk(json!({
                                "delta": {
                                    "reasoning_content": reasoning
                                }
                            })));
                        }
                    }
                    "thinking" => {
                        self.close_text_block(&mut out);
                        self.current_content_block_index = index;
                        self.current_content_block_type = Some("thinking".into());
                    }
                    "tool_use" => {
                        self.close_text_block(&mut out);
                        self.current_content_block_index = index;
                        self.current_content_block_type = Some("tool_use".into());
                        if let Some(i) = index {
                            let id = block
                                .and_then(|b| b.get("id"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = block
                                .and_then(|b| b.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            self.tool_use_states.insert(
                                i,
                                ToolUseState {
                                    id,
                                    name,
                                    ..Default::default()
                                },
                            );
                        }
                    }
                    _ => {}
                }
            }

            Some("content_block_delta") => {
                let index = event
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .map(|i| i as u32);
                let delta = event.get("delta");
                let delta_type = delta.and_then(|d| d.get("type")).and_then(|t| t.as_str());

                match delta_type {
                    Some("text_delta") => {
                        if let Some(text) =
                            delta.and_then(|d| d.get("text")).and_then(|v| v.as_str())
                        {
                            let text = text.to_string();
                            out.push(self.build_chat_chunk(json!({
                                "delta": {
                                    "content": text
                                }
                            })));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(thinking) = delta
                            .and_then(|d| d.get("thinking"))
                            .and_then(|v| v.as_str())
                        {
                            self.pending_reasoning.push_str(thinking);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial_json) = delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|v| v.as_str())
                        {
                            if let Some(i) = index {
                                let (was_new, tid, tname) = {
                                    if let Some(ts) = self.tool_use_states.get_mut(&i) {
                                        let already_started = ts.started;
                                        if !already_started {
                                            ts.started = true;
                                        }
                                        (!already_started, ts.id.clone(), ts.name.clone())
                                    } else {
                                        return out;
                                    }
                                };
                                if was_new {
                                    // Emit reasoning before tool call starts
                                    let reasoning = std::mem::take(&mut self.pending_reasoning);
                                    if !reasoning.is_empty() {
                                        out.push(self.build_chat_chunk(json!({
                                            "delta": {
                                                "reasoning_content": reasoning
                                            }
                                        })));
                                    }
                                    out.push(self.build_chat_chunk(json!({
                                        "delta": {
                                            "tool_calls": [{
                                                "index": i,
                                                "id": tid,
                                                "type": "function",
                                                "function": {
                                                    "name": tname,
                                                    "arguments": ""
                                                }
                                            }]
                                        }
                                    })));
                                }
                                // Append arguments to tool_use state
                                let pj = partial_json.to_string();
                                if let Some(ts) = self.tool_use_states.get_mut(&i) {
                                    ts.arguments.push_str(&pj);
                                }
                                out.push(self.build_chat_chunk(json!({
                                    "delta": {
                                        "tool_calls": [{
                                            "index": i,
                                            "function": {
                                                "arguments": pj
                                            }
                                        }]
                                    }
                                })));
                            }
                        }
                    }
                    _ => {}
                }
            }

            Some("content_block_stop") => {
                let index = event
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .map(|i| i as u32);
                if let Some(ref bt) = self.current_content_block_type {
                    match bt.as_str() {
                        "thinking" => {
                            let reasoning = std::mem::take(&mut self.pending_reasoning);
                            if !reasoning.is_empty() {
                                out.push(self.build_chat_chunk(json!({
                                    "delta": {
                                        "reasoning_content": reasoning
                                    }
                                })));
                            }
                        }
                        "tool_use" => {
                            if let Some(i) = index {
                                if let Some(ts) = self.tool_use_states.get_mut(&i) {
                                    ts.done = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                self.current_content_block_index = None;
                self.current_content_block_type = None;
            }

            Some("message_delta") => {
                if let Some(delta) = event.get("delta") {
                    if let Some(sr) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                        let mapped = map_anthropic_stop_reason(sr);
                        self.finish_reason = Some(mapped.to_string());
                    }
                }
                // Anthropic 只在 message_start 提供完整 usage（含 input_tokens /
                // cache_creation_input_tokens / cache_read_input_tokens）；message_delta
                // 的 usage 通常只带累计的 output_tokens，且缓存字段为 0。因此这里要
                // 合并而非整体覆盖，避免把 message_start 的缓存统计清零。
                if let Some(usage) = event.get("usage").and_then(|u| u.as_object()) {
                    let base = self
                        .usage
                        .get_or_insert_with(|| Value::Object(serde_json::Map::new()));
                    if let Some(base_obj) = base.as_object_mut() {
                        for (k, v) in usage {
                            // 缓存/输入类字段仅在其为非零或 base 尚无该键时才更新，
                            // 防止 delta 里的 0 值覆盖 message_start 的真实统计。
                            let is_prompt_field = k == "input_tokens"
                                || k == "cache_creation_input_tokens"
                                || k == "cache_read_input_tokens";
                            if is_prompt_field {
                                let incoming_zero = v.as_i64().map(|n| n == 0).unwrap_or(false);
                                let base_has =
                                    base_obj.get(k).and_then(|x| x.as_i64()).unwrap_or(0) > 0;
                                if incoming_zero && base_has {
                                    continue;
                                }
                            }
                            base_obj.insert(k.clone(), v.clone());
                        }
                    }
                }
            }

            Some("message_stop") => {
                self.done = true;
                let fr = self.finish_reason.as_deref().unwrap_or("stop");
                let mut chunk = json!({
                    "delta": {},
                    "finish_reason": fr
                });
                if let Some(u) = self.chat_usage() {
                    chunk["usage"] = u;
                }
                out.push(self.build_chat_chunk(chunk));
                // [DONE] is emitted after this returns
            }

            Some("error") => {
                // Anthropic 中途 error 事件（overloaded_error / 余额不足 / 限流等）。
                // 记录原始错误，交由 finalize 向下游转发，避免被合成为正常完成。
                let err_obj = event.get("error");
                let err_type = err_obj
                    .and_then(|e| e.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("error");
                let err_msg = err_obj
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("upstream returned error event");
                log::warn!("[AnthropicToChat] 上游 error 事件: type={err_type}, message={err_msg}");
                self.upstream_error = Some(format!("Upstream error ({err_type}): {err_msg}"));
            }

            Some("ping") => {
                // Keep-alive, ignore.
            }

            _ => {}
        }

        out
    }

    /// 将内部存储的 Anthropic 风格 usage 转换为 OpenAI Chat Completions 格式。
    ///
    /// Anthropic 的 `input_tokens` 不含缓存部分，缓存命中在
    /// `cache_read_input_tokens`、缓存写入在 `cache_creation_input_tokens`；而
    /// OpenAI/Responses 语义里 `prompt_tokens` 应为总输入，缓存命中放在
    /// `prompt_tokens_details.cached_tokens`。这里做等价换算，保证下游能正确
    /// 统计缓存命中率。
    fn chat_usage(&self) -> Option<Value> {
        let u = self.usage.as_ref()?.as_object()?;
        let get = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);

        let raw_input = get("input_tokens");
        let cache_read = get("cache_read_input_tokens");
        let cache_creation = get("cache_creation_input_tokens");
        let output = get("output_tokens");

        // OpenAI prompt_tokens = 全部输入（含缓存读取与缓存写入）。
        let prompt_tokens = raw_input + cache_read + cache_creation;
        let total = prompt_tokens + output;

        Some(json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output,
            "total_tokens": total,
            "prompt_tokens_details": { "cached_tokens": cache_read },
            // 同时保留 Anthropic 原始字段，便于下游透传缓存写入统计。
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation
        }))
    }

    fn build_chat_chunk(&self, chunk: Value) -> Bytes {
        let delta = chunk.get("delta").unwrap_or(&chunk);
        let choice = json!({
            "index": 0,
            "delta": delta,
            "finish_reason": chunk.get("finish_reason")
        });

        let mut full = json!({
            "id": self.chat_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice]
        });
        if let Some(usage) = chunk.get("usage") {
            full["usage"] = usage.clone();
        }

        Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&full).unwrap_or_default()
        ))
    }

    fn close_text_block(&mut self, _out: &mut Vec<Bytes>) {
        // In Chat SSE, text blocks are implicit — no need to emit stop events
        self.current_content_block_index = None;
        self.current_content_block_type = None;
    }
}

fn map_anthropic_stop_reason(reason: &str) -> &str {
    match reason {
        "end_turn" => "stop",
        "max_tokens" => "length",
        "stop_sequence" => "stop",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typeless_error_object_is_captured_as_upstream_error() {
        // 复现 IDE 卡死：JoyCode 内容审查在流里返回无 type 的顶层 error 对象，
        // 旧逻辑落入 `_ => {}` 被静默丢弃，导致流无收尾、客户端一直转圈。
        let mut state = AnthropicToChatState::default();
        let event = json!({
            "error": {
                "cause": "",
                "code": 400,
                "message": "sensitive contain:[\"密码\"]"
            }
        });

        let out = state.handle_anthropic_event(&event);

        assert!(out.is_empty(), "错误事件不应产出正常 chat chunk");
        assert!(
            state
                .upstream_error
                .as_deref()
                .is_some_and(|e| e.contains("sensitive contain")),
            "必须记录 upstream_error，实际: {:?}",
            state.upstream_error
        );
    }

    #[test]
    fn typeless_error_object_prefers_string_type_when_present() {
        // 若 error 对象带字符串 type，则优先采用它。
        let mut state = AnthropicToChatState::default();
        let event = json!({
            "error": {
                "type": "content_filter",
                "message": "blocked"
            }
        });

        state.handle_anthropic_event(&event);

        let err = state.upstream_error.expect("应记录 upstream_error");
        assert!(err.contains("content_filter"), "实际: {err}");
        assert!(err.contains("blocked"));
    }

    #[test]
    fn typeless_event_without_error_is_ignored() {
        // 无 type 且无 error 字段的事件不应误判为错误。
        let mut state = AnthropicToChatState::default();
        let event = json!({ "foo": "bar" });

        let out = state.handle_anthropic_event(&event);

        assert!(out.is_empty());
        assert!(state.upstream_error.is_none());
    }
}

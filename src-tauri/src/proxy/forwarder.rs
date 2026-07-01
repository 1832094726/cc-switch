//! 请求转发器
//!
//! 负责将请求转发到上游Provider，支持故障转移

use super::hyper_client::ProxyResponse;
use super::{
    body_filter::filter_private_params_with_whitelist,
    content_encoding::{decompress_body, get_content_encoding},
    error::*,
    failover_switch::FailoverSwitchManager,
    json_canonical::{
        canonical_json_string, canonicalize_value, short_sha256_hex, short_value_hash,
    },
    log_codes::fwd as log_fwd,
    provider_router::ProviderRouter,
    providers::{
        codex_chat_history::CodexChatHistoryStore, gemini_shadow::GeminiShadowStore, get_adapter,
        AuthInfo, AuthStrategy, JoyCodeAnthropicAdapter, ProviderAdapter, ProviderType,
        JOYCODE_VSCODE_CLIENT, JOYCODE_VSCODE_CLIENT_VERSION,
    },
    thinking_budget_rectifier::{rectify_thinking_budget, should_rectify_thinking_budget},
    thinking_rectifier::{
        normalize_thinking_type, rectify_anthropic_request, should_rectify_thinking_signature,
    },
    types::{CopilotOptimizerConfig, OptimizerConfig, ProxyStatus, RectifierConfig},
    ProxyError,
};
use crate::commands::{CodexOAuthState, CopilotAuthState};
use crate::proxy::providers::codex_oauth_auth::CodexOAuthManager;
use crate::proxy::providers::copilot_auth::CopilotAuthManager;
use crate::{
    app_config::AppType,
    provider::{LocalProxyRequestOverrides, Provider},
};
use bytes::Bytes;
use futures::StreamExt;
use hmac::{Hmac, Mac};
use http::Extensions;
use regex::Regex;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use std::sync::OnceLock;
use tauri::Manager;
use tokio::sync::RwLock;

const PROXY_AUTH_PLACEHOLDER: &str = "PROXY_MANAGED";

pub struct ForwardResult {
    pub response: ProxyResponse,
    pub provider: Provider,
    pub claude_api_format: Option<String>,
    /// 实际发往上游的模型名（路由接管/模型映射后的真值）。
    ///
    /// usage 归因不能依赖 ctx.request_model（映射前的客户端别名）：上游响应
    /// 缺失 model 或回显别名时，接管流量会被记成 claude-* 并按其定价计费。
    pub outbound_model: Option<String>,
    /// 本次上游调用是否把本地 Responses 请求转换成 Chat Completions。
    pub(crate) codex_responses_to_chat: bool,
    /// 本次上游调用是否把本地 Chat Completions 请求转换成 Responses。
    pub(crate) codex_chat_to_responses: bool,
    /// 活跃连接 RAII guard：随响应一起流转到 response_processor / handle_claude_transform，
    /// 最终被 move 进流式 body future（或非流式响应作用域），覆盖整个响应生命周期。
    pub(crate) connection_guard: Option<ActiveConnectionGuard>,
}

pub struct ForwardError {
    pub error: ProxyError,
    pub provider: Option<Provider>,
}

/// 活跃连接 RAII guard
///
/// 构造时把 `ProxyStatus.active_connections` +1；Drop 时在 tokio runtime 上调度
/// 一个异步任务执行 -1，从而支持把 guard move 进流式 body future（stream 自然结束
/// 时 guard 与 future 一起 drop）。
///
/// 设计动机：之前在 `forward_with_retry` 出口处同步 -1，但流式响应的 body 实际
/// 在 `create_logged_passthrough_stream` 内还会继续 yield 字节流，导致 UI 的
/// `active_connections` 计数过早归零。RAII guard 让"减量"由 Rust 类型系统驱动，
/// 不需要每条出口路径都手动调用。
pub(crate) struct ActiveConnectionGuard {
    status: Arc<RwLock<ProxyStatus>>,
}

impl ActiveConnectionGuard {
    pub(crate) async fn acquire(status: Arc<RwLock<ProxyStatus>>) -> Self {
        {
            let mut s = status.write().await;
            s.active_connections = s.active_connections.saturating_add(1);
        }
        Self { status }
    }
}

impl Drop for ActiveConnectionGuard {
    fn drop(&mut self) {
        // Drop 不能 await：把减量操作调度到 tokio runtime
        let status = self.status.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut s = status.write().await;
                s.active_connections = s.active_connections.saturating_sub(1);
            });
        }
        // 没有 runtime 时静默丢失计数（仅 UI 展示用，可接受最终一致性）
    }
}

pub struct RequestForwarder {
    /// 共享的 ProviderRouter（持有熔断器状态）
    router: Arc<ProviderRouter>,
    status: Arc<RwLock<ProxyStatus>>,
    current_providers: Arc<RwLock<std::collections::HashMap<String, (String, String)>>>,
    gemini_shadow: Arc<GeminiShadowStore>,
    codex_chat_history: Arc<CodexChatHistoryStore>,
    /// 故障转移切换管理器
    failover_manager: Arc<FailoverSwitchManager>,
    /// AppHandle，用于发射事件和更新托盘
    app_handle: Option<tauri::AppHandle>,
    /// 请求开始时的"当前供应商 ID"（用于判断是否需要同步 UI/托盘）
    current_provider_id_at_start: String,
    /// 代理会话 ID（用于 Gemini Native shadow replay）
    session_id: String,
    /// Session ID 是否由客户端提供；生成值不能作为上游缓存身份。
    session_client_provided: bool,
    /// 整流器配置
    rectifier_config: RectifierConfig,
    /// 优化器配置
    optimizer_config: OptimizerConfig,
    /// Copilot 优化器配置
    copilot_optimizer_config: CopilotOptimizerConfig,
    /// 非流式请求超时（秒）
    non_streaming_timeout: std::time::Duration,
    /// 流式请求响应头等待超时（秒）
    streaming_first_byte_timeout: std::time::Duration,
    /// 单个客户端请求最多尝试的 provider 数。
    ///
    /// 由 `AppProxyConfig.max_retries` (UI: "请求失败时的重试次数, 0-10") 派生：
    /// `max_attempts = max_retries + 1`，所以 max_retries=0 表示仅尝试一家、
    /// max_retries=3（默认）表示最多 4 家。loop 同时受 providers.len() 自然限制。
    max_attempts: usize,
}

impl RequestForwarder {
    /// 预防式 media 降级：发送前对 text-only 模型把图片块替换为标记。
    ///
    /// 受 `enabled && request_media_fallback` 管辖；其中"启发式模型名单预测"
    /// 再受 `request_media_heuristic` 单独管辖（显式声明 text-only 始终生效）。
    /// 返回被替换的图片块数量（0 = 未触发或开关关闭）。
    fn apply_media_prevention(&self, body: &mut Value, provider: &Provider) -> usize {
        if !(self.rectifier_config.enabled && self.rectifier_config.request_media_fallback) {
            return 0;
        }
        let replaced_images = super::media_sanitizer::replace_images_for_text_only_model(
            body,
            provider,
            self.rectifier_config.request_media_heuristic,
        );
        if replaced_images > 0 {
            let model = body.get("model").and_then(Value::as_str).unwrap_or("");
            log::info!(
                "[Media] Replaced {replaced_images} image block(s) with {} for text-only provider={}, model={}",
                super::media_sanitizer::UNSUPPORTED_IMAGE_MARKER,
                provider.id,
                model
            );
        }
        replaced_images
    }

    /// 反应式 media 重试判定：上游因图片输入报错后，是否应替换图片块并对同一供应商重试一次。
    ///
    /// 受 `enabled && request_media_fallback` 管辖；不涉及 `request_media_heuristic`——
    /// 这里是上游"实测"错误后的纯恢复，不是预测，故启发式开关与它无关。
    fn media_retry_should_trigger(
        &self,
        adapter_name: &str,
        already_retried: bool,
        provider_body: &Value,
        error: &ProxyError,
    ) -> bool {
        matches!(adapter_name, "Claude" | "Codex")
            && self.rectifier_config.enabled
            && self.rectifier_config.request_media_fallback
            && !already_retried
            && super::media_sanitizer::contains_image_blocks(provider_body)
            && super::media_sanitizer::is_unsupported_image_error(error)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: Arc<ProviderRouter>,
        non_streaming_timeout: u64,
        status: Arc<RwLock<ProxyStatus>>,
        current_providers: Arc<RwLock<std::collections::HashMap<String, (String, String)>>>,
        gemini_shadow: Arc<GeminiShadowStore>,
        codex_chat_history: Arc<CodexChatHistoryStore>,
        failover_manager: Arc<FailoverSwitchManager>,
        app_handle: Option<tauri::AppHandle>,
        current_provider_id_at_start: String,
        session_id: String,
        session_client_provided: bool,
        streaming_first_byte_timeout: u64,
        _streaming_idle_timeout: u64,
        rectifier_config: RectifierConfig,
        optimizer_config: OptimizerConfig,
        copilot_optimizer_config: CopilotOptimizerConfig,
        max_retries: u32,
    ) -> Self {
        // max_retries 是「失败后重试次数」语义，attempt 上限 = retries + 1。
        // saturating_add 防止 u32::MAX + 1 溢出。
        let max_attempts = (max_retries as usize).saturating_add(1);
        Self {
            router,
            status,
            current_providers,
            gemini_shadow,
            codex_chat_history,
            failover_manager,
            app_handle,
            current_provider_id_at_start,
            session_id,
            session_client_provided,
            rectifier_config,
            optimizer_config,
            copilot_optimizer_config,
            non_streaming_timeout: std::time::Duration::from_secs(non_streaming_timeout),
            streaming_first_byte_timeout: std::time::Duration::from_secs(
                streaming_first_byte_timeout,
            ),
            max_attempts,
        }
    }

    async fn record_success_result(
        &self,
        provider_id: &str,
        app_type: &str,
        used_half_open_permit: bool,
    ) {
        if used_half_open_permit {
            if let Err(e) = self
                .router
                .record_result(provider_id, app_type, true, true, None)
                .await
            {
                log::warn!(
                    "[{app_type}] 记录 Provider 成功结果失败: provider_id={provider_id}, error={e}"
                );
            }
            return;
        }

        let router = self.router.clone();
        let provider_id = provider_id.to_string();
        let app_type = app_type.to_string();
        tokio::spawn(async move {
            if let Err(e) = router
                .record_result(&provider_id, &app_type, false, true, None)
                .await
            {
                log::warn!(
                    "[{app_type}] 异步记录 Provider 成功结果失败: provider_id={provider_id}, error={e}"
                );
            }
        });
    }

    /// 整流（thinking signature 或 budget）重试失败后的统一收尾。
    ///
    /// `None` 表示已记录熔断器、累积 `last_error`/`last_provider`，
    /// 调用方应 `continue` 让下一家 provider 继续故障转移；
    /// `Some(ForwardError)` 表示是客户端错误，没有 provider 能修复，
    /// 调用方应直接 `return` 把错误返回给客户端。
    #[allow(clippy::too_many_arguments)]
    async fn handle_rectifier_retry_failure(
        &self,
        retry_err: ProxyError,
        provider: &Provider,
        app_type_str: &str,
        used_half_open_permit: bool,
        rectifier_label: &str,
        last_error: &mut Option<ProxyError>,
        last_provider: &mut Option<Provider>,
    ) -> Option<ForwardError> {
        // Provider 错误：本家上游/网络确实出问题，下一家 provider 可能可用 → 继续故障转移。
        // 客户端错误：整流后请求仍违法，下一家也修不好 → 直接返回。
        let is_provider_error = match &retry_err {
            ProxyError::Timeout(_) | ProxyError::ForwardFailed(_) => true,
            ProxyError::UpstreamError { status, .. } => *status >= 500,
            _ => false,
        };

        if is_provider_error {
            let _ = self
                .router
                .record_result(
                    &provider.id,
                    app_type_str,
                    used_half_open_permit,
                    false,
                    Some(retry_err.to_string()),
                )
                .await;
            {
                let mut status = self.status.write().await;
                status.last_error = Some(format!(
                    "Provider {} {rectifier_label}重试失败: {}",
                    provider.name, retry_err
                ));
            }
            *last_error = Some(retry_err);
            *last_provider = Some(provider.clone());
            return None;
        }

        self.router
            .release_permit_neutral(&provider.id, app_type_str, used_half_open_permit)
            .await;
        let mut status = self.status.write().await;
        status.failed_requests += 1;
        status.last_error = Some(retry_err.to_string());
        if status.total_requests > 0 {
            status.success_rate =
                (status.success_requests as f32 / status.total_requests as f32) * 100.0;
        }
        Some(ForwardError {
            error: retry_err,
            provider: Some(provider.clone()),
        })
    }

    /// 转发请求（带故障转移）
    ///
    /// 这是 thin wrapper：在客户端请求维度记一次 `total_requests` / 调整
    /// `active_connections` / 刷新 `last_request_at`，无论 inner 走哪条出口路径，
    /// 出口处都会把 `active_connections` 回收。Per-attempt 维度（成功/失败/熔断
    /// 等）仍由 inner 内自行更新 `success_requests` / `failed_requests`。
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_with_retry(
        &self,
        app_type: &AppType,
        method: http::Method,
        endpoint: &str,
        body: Value,
        headers: axum::http::HeaderMap,
        extensions: Extensions,
        providers: Vec<Provider>,
    ) -> Result<ForwardResult, ForwardError> {
        let guard = ActiveConnectionGuard::acquire(self.status.clone()).await;
        {
            let mut s = self.status.write().await;
            s.total_requests = s.total_requests.saturating_add(1);
            s.last_request_at = Some(chrono::Utc::now().to_rfc3339());
        }
        let result = self
            .forward_with_retry_inner(
                app_type, method, endpoint, body, headers, extensions, providers,
            )
            .await;
        // 把 guard 注入到 Ok 结果，让它随响应一起流转到 response_processor，
        // 在流式 body 的 future 内才真正 drop。
        // Err 路径：guard 在函数 scope 内随返回值落地时自动 drop。
        result.map(|mut fr| {
            fr.connection_guard = Some(guard);
            fr
        })
    }

    /// 实际转发逻辑（不包含客户端维度的入口/出口计数）
    ///
    /// # Arguments
    /// * `app_type` - 应用类型
    /// * `method` - 客户端请求的 HTTP 方法（透传给上游，支持 GET/POST 等）
    /// * `endpoint` - API 端点
    /// * `body` - 请求体
    /// * `headers` - 请求头
    /// * `providers` - 已选择的 Provider 列表（由 RequestContext 提供，避免重复调用 select_providers）
    #[allow(clippy::too_many_arguments)]
    async fn forward_with_retry_inner(
        &self,
        app_type: &AppType,
        method: http::Method,
        endpoint: &str,
        body: Value,
        headers: axum::http::HeaderMap,
        extensions: Extensions,
        providers: Vec<Provider>,
    ) -> Result<ForwardResult, ForwardError> {
        // 获取适配器
        let adapter = get_adapter(app_type);
        let app_type_str = app_type.as_str();

        if providers.is_empty() {
            return Err(ForwardError {
                error: ProxyError::NoAvailableProvider,
                provider: None,
            });
        }

        let mut last_error = None;
        let mut last_provider = None;
        let mut attempted_providers = 0usize;

        // 单 Provider 场景下跳过熔断器检查（故障转移关闭时）
        let bypass_circuit_breaker = providers.len() == 1;

        // 依次尝试每个供应商
        for provider in providers.iter() {
            // 整流器重试标记：每个 provider 独立持有，避免标记跨 provider 短路故障转移
            // —— 首家 provider 整流后被 5xx/timeout 击落时，下家仍能用整流后的请求体走整流流程
           let mut rectifier_retried = false;
           let mut budget_rectifier_retried = false;
           let mut media_rectifier_retried = false;
            // 同供应商超时重试计数器：首包超时通常是瞬时网络问题，最多重试 3 次
            let mut timeout_retries = 0u32;
            const MAX_TIMEOUT_RETRIES: u32 = 3;

            // 上限检查：尊重用户在 AppProxyConfig.max_retries 上配置的「重试次数」。
            // 放在熔断器 allow 检查之前，避免在已经超限时还占用 HalfOpen 探测名额。
            if attempted_providers >= self.max_attempts {
                log::warn!(
                    "[{app_type_str}] 已达最大尝试次数上限 ({}/{}), 停止故障转移",
                    attempted_providers,
                    self.max_attempts
                );
                break;
            }

            // 发起请求前先获取熔断器放行许可（HalfOpen 会占用探测名额）
            // 单 Provider 场景下跳过此检查，避免熔断器阻塞所有请求
            let (allowed, used_half_open_permit) = if bypass_circuit_breaker {
                (true, false)
            } else {
                let permit = self
                    .router
                    .allow_provider_request(&provider.id, app_type_str)
                    .await;
                (permit.allowed, permit.used_half_open_permit)
            };

            if !allowed {
                continue;
            }

            // PRE-SEND 优化器：每个 provider 独立决定是否优化
            // clone body 以避免 Bedrock 优化字段泄漏到非 Bedrock provider（failover 场景）
            let mut provider_body =
                if self.optimizer_config.enabled && is_bedrock_provider(provider) {
                    let mut b = body.clone();
                    if self.optimizer_config.thinking_optimizer {
                        super::thinking_optimizer::optimize(&mut b, &self.optimizer_config);
                    }
                    if self.optimizer_config.cache_injection {
                        super::cache_injector::inject(&mut b, &self.optimizer_config);
                    }
                    b
                } else {
                    body.clone()
                };

            attempted_providers += 1;

            // 更新状态中的当前 Provider 信息（per-attempt 维度的标识）
            //
            // total_requests / last_request_at / active_connections 已由
            // forward_with_retry wrapper 在客户端请求维度统一处理，这里只刷
            // 新「正在尝试哪个 provider」的展示字段。
            {
                let mut status = self.status.write().await;
                status.current_provider = Some(provider.name.clone());
                status.current_provider_id = Some(provider.id.clone());
            }

            // 转发请求（每个 Provider 只尝试一次，重试由客户端控制）
            match self
                .forward(
                    app_type,
                    &method,
                    provider,
                    endpoint,
                    &provider_body,
                    &headers,
                    &extensions,
                    adapter.as_ref(),
                )
                .await
            {
                Ok((
                    response,
                    claude_api_format,
                    outbound_model,
                    codex_responses_to_chat,
                    codex_chat_to_responses,
                )) => {
                    // 成功：普通闭合熔断状态异步记录，避免阻塞流式首包返回；
                    // HalfOpen 探测仍同步等待，保证 permit 与熔断状态及时释放。
                    self.record_success_result(&provider.id, app_type_str, used_half_open_permit)
                        .await;

                    // 更新当前应用类型使用的 provider
                    {
                        let mut current_providers = self.current_providers.write().await;
                        current_providers.insert(
                            app_type_str.to_string(),
                            (provider.id.clone(), provider.name.clone()),
                        );
                    }

                    // 更新成功统计
                    {
                        let mut status = self.status.write().await;
                        status.success_requests += 1;
                        status.last_error = None;
                        let should_switch =
                            self.current_provider_id_at_start.as_str() != provider.id.as_str();
                        if should_switch {
                            status.failover_count += 1;

                            // 异步触发供应商切换，更新 UI/托盘，并把“当前供应商”同步为实际使用的 provider
                            let fm = self.failover_manager.clone();
                            let ah = self.app_handle.clone();
                            let pid = provider.id.clone();
                            let pname = provider.name.clone();
                            let at = app_type_str.to_string();

                            tokio::spawn(async move {
                                let _ = fm.try_switch(ah.as_ref(), &at, &pid, &pname).await;
                            });
                        }
                        // 重新计算成功率
                        if status.total_requests > 0 {
                            status.success_rate = (status.success_requests as f32
                                / status.total_requests as f32)
                                * 100.0;
                        }
                    }

                    return Ok(ForwardResult {
                        response,
                        provider: provider.clone(),
                        claude_api_format,
                        outbound_model,
                        codex_responses_to_chat,
                        codex_chat_to_responses,
                        connection_guard: None,
                    });
                }
                Err(e) => {
                    // 检测是否需要触发整流器（仅 Claude/ClaudeAuth 供应商）
                    let provider_type = ProviderType::from_app_type_and_config(app_type, provider);
                    let is_anthropic_provider = matches!(
                        provider_type,
                        ProviderType::Claude | ProviderType::ClaudeAuth
                    );
                    let mut signature_rectifier_non_retryable_client_error = false;

                    if self.media_retry_should_trigger(
                        adapter.name(),
                        media_rectifier_retried,
                        &provider_body,
                        &e,
                    ) {
                        let mut media_body = provider_body.clone();
                        let replaced_images =
                            super::media_sanitizer::replace_image_blocks_with_marker(
                                &mut media_body,
                            );

                        if replaced_images > 0 {
                            let _ = std::mem::replace(&mut media_rectifier_retried, true);
                            let model = media_body
                                .get("model")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            log::info!(
                                "[{app_type_str}] [Media] Upstream rejected image input; retrying provider={} model={} with {replaced_images} image block(s) replaced by {}",
                                provider.id,
                                model,
                                super::media_sanitizer::UNSUPPORTED_IMAGE_MARKER
                            );

                            match self
                                .forward(
                                    app_type,
                                    &method,
                                    provider,
                                    endpoint,
                                    &media_body,
                                    &headers,
                                    &extensions,
                                    adapter.as_ref(),
                                )
                                .await
                            {
                                Ok((
                                    response,
                                    claude_api_format,
                                    outbound_model,
                                    codex_responses_to_chat,
                                    codex_chat_to_responses,
                                )) => {
                                    log::info!(
                                        "[{app_type_str}] [Media] Unsupported-image retry succeeded"
                                    );
                                    self.record_success_result(
                                        &provider.id,
                                        app_type_str,
                                        used_half_open_permit,
                                    )
                                    .await;

                                    {
                                        let mut current_providers =
                                            self.current_providers.write().await;
                                        current_providers.insert(
                                            app_type_str.to_string(),
                                            (provider.id.clone(), provider.name.clone()),
                                        );
                                    }

                                    {
                                        let mut status = self.status.write().await;
                                        status.success_requests += 1;
                                        status.last_error = None;
                                        let should_switch =
                                            self.current_provider_id_at_start.as_str()
                                                != provider.id.as_str();
                                        if should_switch {
                                            status.failover_count += 1;
                                            let fm = self.failover_manager.clone();
                                            let ah = self.app_handle.clone();
                                            let pid = provider.id.clone();
                                            let pname = provider.name.clone();
                                            let at = app_type_str.to_string();

                                            tokio::spawn(async move {
                                                let _ = fm
                                                    .try_switch(ah.as_ref(), &at, &pid, &pname)
                                                    .await;
                                            });
                                        }
                                        if status.total_requests > 0 {
                                            status.success_rate = (status.success_requests as f32
                                                / status.total_requests as f32)
                                                * 100.0;
                                        }
                                    }

                                    return Ok(ForwardResult {
                                        response,
                                        provider: provider.clone(),
                                        claude_api_format,
                                        outbound_model,
                                        codex_responses_to_chat,
                                        codex_chat_to_responses,
                                        connection_guard: None,
                                    });
                                }
                                Err(retry_err) => {
                                    log::warn!(
                                        "[{app_type_str}] [Media] Unsupported-image retry still failed: {retry_err}"
                                    );
                                    if let Some(err) = self
                                        .handle_rectifier_retry_failure(
                                            retry_err,
                                            provider,
                                            app_type_str,
                                            used_half_open_permit,
                                            "media 降级",
                                            &mut last_error,
                                            &mut last_provider,
                                        )
                                        .await
                                    {
                                        return Err(err);
                                    }
                                    continue;
                                }
                            }
                        }
                    }

                    if is_anthropic_provider {
                        let error_message = extract_error_message(&e);
                        if should_rectify_thinking_signature(
                            error_message.as_deref(),
                            &self.rectifier_config,
                        ) {
                            // 已经重试过：直接返回错误（不可重试客户端错误）
                            if rectifier_retried {
                                log::warn!("[{app_type_str}] [RECT-005] 整流器已触发过，不再重试");
                                // 释放 HalfOpen permit（不记录熔断器，这是客户端兼容性问题）
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type_str,
                                        used_half_open_permit,
                                    )
                                    .await;
                                let mut status = self.status.write().await;
                                status.failed_requests += 1;
                                status.last_error = Some(e.to_string());
                                if status.total_requests > 0 {
                                    status.success_rate = (status.success_requests as f32
                                        / status.total_requests as f32)
                                        * 100.0;
                                }
                                return Err(ForwardError {
                                    error: e,
                                    provider: Some(provider.clone()),
                                });
                            }

                            // 首次触发：整流请求体
                            let rectified = rectify_anthropic_request(&mut provider_body);

                            // 整流未生效：继续尝试 budget 整流路径，避免误判后短路
                            if !rectified.applied {
                                log::warn!(
                                    "[{app_type_str}] [RECT-006] thinking 签名整流器触发但无可整流内容，继续检查 budget；若 budget 也未命中则按客户端错误返回"
                                );
                                signature_rectifier_non_retryable_client_error = true;
                            } else {
                                log::info!(
                                    "[{}] [RECT-001] thinking 签名整流器触发, 移除 {} thinking blocks, {} redacted_thinking blocks, {} signature fields",
                                    app_type_str,
                                    rectified.removed_thinking_blocks,
                                    rectified.removed_redacted_thinking_blocks,
                                    rectified.removed_signature_fields
                                );

                                // 标记已重试（当前逻辑下重试后必定 return，保留标记以备将来扩展）
                                let _ = std::mem::replace(&mut rectifier_retried, true);

                                // 使用同一供应商重试（不计入熔断器）
                                match self
                                    .forward(
                                        app_type,
                                        &method,
                                        provider,
                                        endpoint,
                                        &provider_body,
                                        &headers,
                                        &extensions,
                                        adapter.as_ref(),
                                    )
                                    .await
                                {
                                    Ok((
                                        response,
                                        claude_api_format,
                                        outbound_model,
                                        codex_responses_to_chat,
                                        codex_chat_to_responses,
                                    )) => {
                                        log::info!("[{app_type_str}] [RECT-002] 整流重试成功");
                                        self.record_success_result(
                                            &provider.id,
                                            app_type_str,
                                            used_half_open_permit,
                                        )
                                        .await;

                                        // 更新当前应用类型使用的 provider
                                        {
                                            let mut current_providers =
                                                self.current_providers.write().await;
                                            current_providers.insert(
                                                app_type_str.to_string(),
                                                (provider.id.clone(), provider.name.clone()),
                                            );
                                        }

                                        // 更新成功统计
                                        {
                                            let mut status = self.status.write().await;
                                            status.success_requests += 1;
                                            status.last_error = None;
                                            let should_switch =
                                                self.current_provider_id_at_start.as_str()
                                                    != provider.id.as_str();
                                            if should_switch {
                                                status.failover_count += 1;

                                                // 异步触发供应商切换，更新 UI/托盘
                                                let fm = self.failover_manager.clone();
                                                let ah = self.app_handle.clone();
                                                let pid = provider.id.clone();
                                                let pname = provider.name.clone();
                                                let at = app_type_str.to_string();

                                                tokio::spawn(async move {
                                                    let _ = fm
                                                        .try_switch(ah.as_ref(), &at, &pid, &pname)
                                                        .await;
                                                });
                                            }
                                            if status.total_requests > 0 {
                                                status.success_rate = (status.success_requests
                                                    as f32
                                                    / status.total_requests as f32)
                                                    * 100.0;
                                            }
                                        }

                                        return Ok(ForwardResult {
                                            response,
                                            provider: provider.clone(),
                                            claude_api_format,
                                            outbound_model,
                                            codex_responses_to_chat,
                                            codex_chat_to_responses,
                                            connection_guard: None,
                                        });
                                    }
                                    Err(retry_err) => {
                                        log::warn!(
                                            "[{app_type_str}] [RECT-003] 整流重试仍失败: {retry_err}"
                                        );
                                        if let Some(err) = self
                                            .handle_rectifier_retry_failure(
                                                retry_err,
                                                provider,
                                                app_type_str,
                                                used_half_open_permit,
                                                "整流",
                                                &mut last_error,
                                                &mut last_provider,
                                            )
                                            .await
                                        {
                                            return Err(err);
                                        }
                                        continue;
                                    }
                                }
                            }
                        }
                    }

                    // 检测是否需要触发 budget 整流器（仅 Claude/ClaudeAuth 供应商）
                    if is_anthropic_provider {
                        let error_message = extract_error_message(&e);
                        if should_rectify_thinking_budget(
                            error_message.as_deref(),
                            &self.rectifier_config,
                        ) {
                            // 已经重试过：直接返回错误（不可重试客户端错误）
                            if budget_rectifier_retried {
                                log::warn!(
                                    "[{app_type_str}] [RECT-013] budget 整流器已触发过，不再重试"
                                );
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type_str,
                                        used_half_open_permit,
                                    )
                                    .await;
                                let mut status = self.status.write().await;
                                status.failed_requests += 1;
                                status.last_error = Some(e.to_string());
                                if status.total_requests > 0 {
                                    status.success_rate = (status.success_requests as f32
                                        / status.total_requests as f32)
                                        * 100.0;
                                }
                                return Err(ForwardError {
                                    error: e,
                                    provider: Some(provider.clone()),
                                });
                            }

                            let budget_rectified = rectify_thinking_budget(&mut provider_body);
                            if !budget_rectified.applied {
                                log::warn!(
                                    "[{app_type_str}] [RECT-014] budget 整流器触发但无可整流内容，不做无意义重试"
                                );
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type_str,
                                        used_half_open_permit,
                                    )
                                    .await;
                                let mut status = self.status.write().await;
                                status.failed_requests += 1;
                                status.last_error = Some(e.to_string());
                                if status.total_requests > 0 {
                                    status.success_rate = (status.success_requests as f32
                                        / status.total_requests as f32)
                                        * 100.0;
                                }
                                return Err(ForwardError {
                                    error: e,
                                    provider: Some(provider.clone()),
                                });
                            }

                            log::info!(
                                "[{}] [RECT-010] thinking budget 整流器触发, before={:?}, after={:?}",
                                app_type_str,
                                budget_rectified.before,
                                budget_rectified.after
                            );

                            let _ = std::mem::replace(&mut budget_rectifier_retried, true);

                            // 使用同一供应商重试（不计入熔断器）
                            match self
                                .forward(
                                    app_type,
                                    &method,
                                    provider,
                                    endpoint,
                                    &provider_body,
                                    &headers,
                                    &extensions,
                                    adapter.as_ref(),
                                )
                                .await
                            {
                                Ok((
                                    response,
                                    claude_api_format,
                                    outbound_model,
                                    codex_responses_to_chat,
                                    codex_chat_to_responses,
                                )) => {
                                    log::info!("[{app_type_str}] [RECT-011] budget 整流重试成功");
                                    self.record_success_result(
                                        &provider.id,
                                        app_type_str,
                                        used_half_open_permit,
                                    )
                                    .await;

                                    {
                                        let mut current_providers =
                                            self.current_providers.write().await;
                                        current_providers.insert(
                                            app_type_str.to_string(),
                                            (provider.id.clone(), provider.name.clone()),
                                        );
                                    }

                                    {
                                        let mut status = self.status.write().await;
                                        status.success_requests += 1;
                                        status.last_error = None;
                                        let should_switch =
                                            self.current_provider_id_at_start.as_str()
                                                != provider.id.as_str();
                                        if should_switch {
                                            status.failover_count += 1;
                                            let fm = self.failover_manager.clone();
                                            let ah = self.app_handle.clone();
                                            let pid = provider.id.clone();
                                            let pname = provider.name.clone();
                                            let at = app_type_str.to_string();
                                            tokio::spawn(async move {
                                                let _ = fm
                                                    .try_switch(ah.as_ref(), &at, &pid, &pname)
                                                    .await;
                                            });
                                        }
                                        if status.total_requests > 0 {
                                            status.success_rate = (status.success_requests as f32
                                                / status.total_requests as f32)
                                                * 100.0;
                                        }
                                    }

                                    return Ok(ForwardResult {
                                        response,
                                        provider: provider.clone(),
                                        claude_api_format,
                                        outbound_model,
                                        codex_responses_to_chat,
                                        codex_chat_to_responses,
                                        connection_guard: None,
                                    });
                                }
                                Err(retry_err) => {
                                    log::warn!(
                                        "[{app_type_str}] [RECT-012] budget 整流重试仍失败: {retry_err}"
                                    );
                                    if let Some(err) = self
                                        .handle_rectifier_retry_failure(
                                            retry_err,
                                            provider,
                                            app_type_str,
                                            used_half_open_permit,
                                            "budget 整流",
                                            &mut last_error,
                                            &mut last_provider,
                                        )
                                        .await
                                    {
                                        return Err(err);
                                    }
                                    continue;
                                }
                            }
                        }
                    }

                    if signature_rectifier_non_retryable_client_error {
                        self.router
                            .release_permit_neutral(
                                &provider.id,
                                app_type_str,
                                used_half_open_permit,
                            )
                            .await;
                        let mut status = self.status.write().await;
                        status.failed_requests += 1;
                        status.last_error = Some(e.to_string());
                        if status.total_requests > 0 {
                            status.success_rate = (status.success_requests as f32
                                / status.total_requests as f32)
                                * 100.0;
                        }
                        return Err(ForwardError {
                            error: e,
                            provider: Some(provider.clone()),
                        });
                    }

                    // 先分类错误，决定是否计入 provider 健康度
                    // —— NonRetryable / ClientAbort 是客户端层错误，无论换哪家 provider 都会被拒绝，
                    //    不应污染熔断器和数据库健康度（与 release_permit_neutral 同语义）。
                    let category = self.categorize_proxy_error(&e);

                    match category {
                        ErrorCategory::Retryable => {
                            // 可重试：真正的 provider 故障 → 记录失败并更新熔断器/DB 健康度
                            let _ = self
                                .router
                                .record_result(
                                    &provider.id,
                                    app_type_str,
                                    used_half_open_permit,
                                    false,
                                    Some(e.to_string()),
                                )
                                .await;

                            {
                                let mut status = self.status.write().await;
                                status.last_error =
                                    Some(format!("Provider {} 失败: {}", provider.name, e));
                            }

                            let (log_code, log_message) = build_retryable_failure_log(
                                &provider.name,
                                attempted_providers,
                                providers.len(),
                                &e,
                            );
                           log::warn!("[{app_type_str}] [{log_code}] {log_message}");

                           // 首包超时通常是瞬时网络问题，重试同一供应商最多 3 次
                           while timeout_retries < MAX_TIMEOUT_RETRIES && matches!(e, ProxyError::Timeout(_)) {
                               timeout_retries += 1;
                               log::info!(
                                   "[{app_type_str}] [FWD-011] 首包超时，重试同一供应商 {} (第 {}/{})",
                                   provider.name, timeout_retries, MAX_TIMEOUT_RETRIES
                               );

                               // 释放当前 permit（避免 HalfOpen 名额泄漏）
                               self.router
                                   .release_permit_neutral(
                                       &provider.id,
                                       app_type_str,
                                       used_half_open_permit,
                                   )
                                   .await;

                               match self
                                   .forward(
                                       app_type,
                                       &method,
                                       provider,
                                       endpoint,
                                       &provider_body,
                                       &headers,
                                       &extensions,
                                       adapter.as_ref(),
                                   )
                                   .await
                               {
                                   Ok((
                                       response,
                                       claude_api_format,
                                       outbound_model,
                                       codex_responses_to_chat,
                                       codex_chat_to_responses,
                                   )) => {
                                       log::info!(
                                           "[{app_type_str}] [FWD-012] 超时重试成功: {}",
                                           provider.name
                                       );
                                       self.record_success_result(
                                           &provider.id,
                                           app_type_str,
                                           false,
                                       )
                                       .await;

                                       {
                                           let mut current_providers =
                                               self.current_providers.write().await;
                                           current_providers.insert(
                                               app_type_str.to_string(),
                                               (provider.id.clone(), provider.name.clone()),
                                           );
                                       }

                                       {
                                           let mut status = self.status.write().await;
                                           status.success_requests += 1;
                                           status.last_error = None;
                                       }

                                       return Ok(ForwardResult {
                                           response,
                                           provider: provider.clone(),
                                           claude_api_format,
                                           outbound_model,
                                           codex_responses_to_chat,
                                           codex_chat_to_responses,
                                           connection_guard: None,
                                       });
                                   }
                                   Err(retry_err) => {
                                       log::warn!(
                                           "[{app_type_str}] [FWD-013] 超时重试第 {} 次仍失败: {retry_err}",
                                           timeout_retries
                                       );
                                       if !matches!(retry_err, ProxyError::Timeout(_)) {
                                           last_error = Some(retry_err);
                                           last_provider = Some(provider.clone());
                                           break;
                                       }
                                   }
                               }
                           }

                           // 超时重试已执行：last_error 已在循环内设置（非超时错误）或需回退到原始错误
                           if timeout_retries > 0 {
                               if last_error.is_none() {
                                   last_error = Some(e);
                               }
                               if last_provider.is_none() {
                                   last_provider = Some(provider.clone());
                               }
                               continue;
                           }

                            last_error = Some(e);
                            last_provider = Some(provider.clone());
                            // 继续尝试下一个供应商
                            continue;
                        }
                       ErrorCategory::FailoverNeutral => {
                            // 模型/端点级错误（如 404）：触发故障转移到下一家供应商，
                            // 但不污染熔断器健康度——该供应商对其他模型/端点可能完全正常。
                            self.router
                                .release_permit_neutral(
                                    &provider.id,
                                    app_type_str,
                                    used_half_open_permit,
                                )
                                .await;
                            {
                                let mut status = self.status.write().await;
                                status.last_error = Some(format!(
                                    "Provider {} 模型/端点不可用: {}",
                                    provider.name, e
                                ));
                            }
                            log::warn!(
                                "[{app_type_str}] [FWD-009] Provider {} 返回模型/端点级错误（404），\
                                 故障转移到下一家，不计入熔断器",
                                provider.name
                            );
                            last_error = Some(e);
                            last_provider = Some(provider.clone());
                            continue;
                        }
                       ErrorCategory::NonRetryable | ErrorCategory::ClientAbort => {
                            // 不可重试：客户端层错误或客户端断连 → 不污染健康度，仅释放 HalfOpen permit
                            self.router
                                .release_permit_neutral(
                                    &provider.id,
                                    app_type_str,
                                    used_half_open_permit,
                                )
                                .await;
                            {
                                let mut status = self.status.write().await;
                                status.failed_requests += 1;
                                status.last_error = Some(e.to_string());
                                if status.total_requests > 0 {
                                    status.success_rate = (status.success_requests as f32
                                        / status.total_requests as f32)
                                        * 100.0;
                                }
                            }
                            return Err(ForwardError {
                                error: e,
                                provider: Some(provider.clone()),
                            });
                        }
                    }
                }
            }
        }

        if attempted_providers == 0 {
            // providers 列表非空，但全部被熔断器拒绝（典型：HalfOpen 探测名额被占用）
            {
                let mut status = self.status.write().await;
                status.failed_requests += 1;
                status.last_error = Some("所有供应商暂时不可用（熔断器限制）".to_string());
                if status.total_requests > 0 {
                    status.success_rate =
                        (status.success_requests as f32 / status.total_requests as f32) * 100.0;
                }
            }
            return Err(ForwardError {
                error: ProxyError::NoAvailableProvider,
                provider: None,
            });
        }

        // 所有供应商都失败了
        {
            let mut status = self.status.write().await;
            status.failed_requests += 1;
            status.last_error = Some("所有供应商都失败".to_string());
            if status.total_requests > 0 {
                status.success_rate =
                    (status.success_requests as f32 / status.total_requests as f32) * 100.0;
            }
        }

        if let Some((log_code, log_message)) =
            build_terminal_failure_log(attempted_providers, providers.len(), last_error.as_ref())
        {
            log::warn!("[{app_type_str}] [{log_code}] {log_message}");
        }

        Err(ForwardError {
            error: last_error.unwrap_or(ProxyError::MaxRetriesExceeded),
            provider: last_provider,
        })
    }

    /// 转发单个请求（使用适配器）
    ///
    /// 成功时返回 `(response, claude_api_format, outbound_model, codex_responses_to_chat, codex_chat_to_responses)`。
    /// 其中 `outbound_model` 是最终发往上游的模型名（所有映射/改写之后）。
    #[allow(clippy::too_many_arguments)]
    async fn forward(
        &self,
        app_type: &AppType,
        method: &http::Method,
        provider: &Provider,
        endpoint: &str,
        body: &Value,
        headers: &axum::http::HeaderMap,
        extensions: &Extensions,
        adapter: &dyn ProviderAdapter,
    ) -> Result<(ProxyResponse, Option<String>, Option<String>, bool, bool), ProxyError> {
        // 使用适配器提取 base_url
        let mut base_url = adapter.extract_base_url(provider)?;

        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);

        // GitHub Copilot API 使用 /chat/completions（无 /v1 前缀）
        let is_copilot = provider
            .meta
            .as_ref()
            .and_then(|m| m.provider_type.as_deref())
            == Some("github_copilot")
            || base_url.contains("githubcopilot.com");

        // 应用模型映射（独立于格式转换）
        // Claude Desktop proxy 模式必须先把 Desktop 可见的 claude-* route
        // 映射成真实上游模型名，并且未知 route 要直接报错，不能使用默认模型兜底。
        let mapped_body = if matches!(app_type, AppType::ClaudeDesktop) {
            crate::claude_desktop_config::map_proxy_request_model(body.clone(), provider)
                .map_err(|e| ProxyError::InvalidRequest(e.to_string()))?
        } else {
            let (mapped_body, _original_model, _mapped_model) =
                super::model_mapper::apply_model_mapping(body.clone(), provider);
            mapped_body
        };

        // 与 CCH 对齐：请求前不做 thinking 主动改写（仅保留兼容入口）
        let mut mapped_body = normalize_thinking_type(mapped_body);

        // 全局清除历史对话中残留的 <cc-switch:thinking> 文本标签。
        // 旧版本生成的这些标签会缓存在客户端对话历史中并被回放，
        // 导致 thinking 内容泄漏到可见对话内容。
        mapped_body = super::copilot_optimizer::strip_thinking_blocks(mapped_body);

        if is_copilot {
            mapped_body =
                super::providers::copilot_model_map::apply_copilot_model_normalization(mapped_body);
            self.apply_copilot_live_model_resolution(provider, &mut mapped_body)
                .await;
        } else {
            mapped_body =
                super::model_mapper::strip_one_m_suffix_for_upstream_from_body(mapped_body);
        }

        let model_catalog_route = if matches!(app_type, AppType::Devin) {
            None
        } else {
            resolve_model_catalog_route(provider, &mapped_body)
        };

        if let Some(route) = model_catalog_route.as_ref() {
            if let Some(route_base_url) = route.base_url.as_deref() {
                base_url = route_base_url.to_string();
            }
            if let Some(upstream_model) = route.upstream_model.as_deref() {
                mapped_body["model"] = Value::String(upstream_model.to_string());
            }
            log::debug!(
                "[ModelCatalog] Route: requested={} upstream={} endpoint={} base_url={}",
                route.requested_model.as_deref().unwrap_or("<none>"),
                route.upstream_model.as_deref().unwrap_or("<same>"),
                route.endpoint.as_deref().unwrap_or("<provider>"),
                route.base_url.as_deref().unwrap_or("<provider>")
            );
        }

        // --- Copilot 优化器：分类 + 请求体优化（在格式转换之前执行） ---
        // 注意：确定性 ID 也在此处计算，因为 mapped_body 在格式转换时会被 move
        //
        // 执行顺序（与 copilot-api 对齐）：
        //   1. 先在原始 body 上分类（保留 tool_result 语义，避免误判为 user）
        //   2. 再清洗孤立 tool_result（防止上游 API 报错）
        //   3. 再合并 tool_result + text（减少 premium 计费）
        let copilot_optimization = if is_copilot && self.copilot_optimizer_config.enabled {
            // 1. 在原始 body 上分类 — 必须在清洗/合并之前执行
            //    孤立 tool_result 仍保持 tool_result 类型，分类能正确识别为 agent
            let has_anthropic_beta = headers.contains_key("anthropic-beta");
            let classification = super::copilot_optimizer::classify_request(
                &mapped_body,
                has_anthropic_beta,
                self.copilot_optimizer_config.compact_detection,
                self.copilot_optimizer_config.subagent_detection,
            );

            log::debug!(
                "[Copilot] 优化器分类: initiator={}, is_warmup={}, is_compact={}, is_subagent={}",
                classification.initiator,
                classification.is_warmup,
                classification.is_compact,
                classification.is_subagent
            );

            // 2. 孤立 tool_result 清理 — 分类完成后再清洗
            //    防止上游 API 因不匹配的 tool_result 报错导致重试/重复计费
            mapped_body = super::copilot_optimizer::sanitize_orphan_tool_results(mapped_body);

            // 3. Tool result 合并 — 将 [tool_result, text] 变为 [tool_result(含text)]
            if self.copilot_optimizer_config.tool_result_merging {
                mapped_body = super::copilot_optimizer::merge_tool_results(mapped_body);
            }

            // 3.5. 主动剥离 thinking block — Copilot 走 OpenAI 兼容端点不识别该块
            //      避免上游拒绝后由 rectifier 反应式重试（首次请求已消耗 quota）
            if self.copilot_optimizer_config.strip_thinking {
                mapped_body = super::copilot_optimizer::strip_thinking_blocks(mapped_body);
            }

            // 4. Warmup 小模型降级
            if self.copilot_optimizer_config.warmup_downgrade && classification.is_warmup {
                log::info!(
                    "[Copilot] Warmup 请求降级到模型: {}",
                    self.copilot_optimizer_config.warmup_model
                );
                mapped_body["model"] =
                    serde_json::json!(&self.copilot_optimizer_config.warmup_model);
            }

            // 预计算确定性 Request ID（在 body 被 move 之前）
            // Session 提取优先级（与 session.rs extract_from_metadata 对齐）：
            //   1. metadata.user_id 中的 _session_ 后缀
            //   2. metadata.session_id（直接字段）
            //   3. raw metadata.user_id（整串 fallback）
            //   4. x-session-id header
            let metadata = body.get("metadata");
            let session_id = metadata
                .and_then(|m| m.get("user_id"))
                .and_then(|v| v.as_str())
                .and_then(super::session::parse_session_from_user_id)
                .or_else(|| {
                    metadata
                        .and_then(|m| m.get("session_id"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                })
                .or_else(|| {
                    metadata
                        .and_then(|m| m.get("user_id"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                })
                .or_else(|| {
                    headers
                        .get("x-session-id")
                        .and_then(|v| v.to_str().ok())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                })
                .unwrap_or_default();
            let det_request_id = if self.copilot_optimizer_config.deterministic_request_id {
                Some(super::copilot_optimizer::deterministic_request_id(
                    &mapped_body,
                    &session_id,
                ))
            } else {
                None
            };

            // 从 session ID 派生稳定的 interaction ID（同一主对话共享）
            let interaction_id =
                super::copilot_optimizer::deterministic_interaction_id(&session_id);

            Some((classification, det_request_id, interaction_id))
        } else {
            None
        };

        // GitHub Copilot 动态 endpoint 路由
        // 从 CopilotAuthManager 获取缓存的 API endpoint（支持企业版等非默认 endpoint）
        if is_copilot && !is_full_url {
            if let Some(app_handle) = &self.app_handle {
                let copilot_state = app_handle.state::<CopilotAuthState>();
                let copilot_auth = copilot_state.0.read().await;

                // 从 provider.meta 获取关联的 GitHub 账号 ID
                let account_id = provider
                    .meta
                    .as_ref()
                    .and_then(|m| m.managed_account_id_for("github_copilot"));

                let dynamic_endpoint = match &account_id {
                    Some(id) => copilot_auth.get_api_endpoint(id).await,
                    None => copilot_auth.get_default_api_endpoint().await,
                };

                // 只在动态 endpoint 与当前 base_url 不同时替换
                if dynamic_endpoint != base_url {
                    log::debug!(
                        "[Copilot] 使用动态 API endpoint: {} (原: {})",
                        dynamic_endpoint,
                        base_url
                    );
                    base_url = dynamic_endpoint;
                }
            }
        }
        let resolved_claude_api_format = if adapter.name() == "Claude" {
            if let Some(api_format) = model_catalog_route
                .as_ref()
                .and_then(|route| route.api_format.clone())
            {
                Some(api_format)
            } else {
                Some(
                    self.resolve_claude_api_format(provider, &mapped_body, is_copilot)
                        .await,
                )
            }
        } else {
            None
        };
        if adapter.name() == "Claude" {
            if let Some(api_format) = resolved_claude_api_format.as_deref() {
                super::providers::normalize_anthropic_messages_for_provider(
                    &mut mapped_body,
                    provider,
                    api_format,
                );
                self.apply_media_prevention(&mut mapped_body, provider);
            }
        }
        let devin_route = if matches!(app_type, AppType::Devin) {
            resolve_devin_model_route(provider, &mapped_body)
        } else {
            None
        };

        if let Some(route) = devin_route.as_ref() {
            if let Some(route_base_url) = route.base_url.as_deref() {
                base_url = route_base_url.to_string();
            }
            if let Some(upstream_model) = route.upstream_model.as_deref() {
                mapped_body["model"] = Value::String(upstream_model.to_string());
            }
            log::debug!(
                "[Devin] Model route: requested={} upstream={} endpoint={} route={}",
                route.requested_model.as_deref().unwrap_or("<none>"),
                route.upstream_model.as_deref().unwrap_or("<same>"),
                route.endpoint,
                route.name.as_deref().unwrap_or("default")
            );
        }

        let devin_upstream_is_messages = devin_route
            .as_ref()
            .is_some_and(|route| is_messages_endpoint(&route.endpoint));
        let devin_upstream_is_chat = devin_route
            .as_ref()
            .is_some_and(|route| is_chat_completions_endpoint(&route.endpoint));
        let devin_upstream_is_responses = devin_route
            .as_ref()
            .is_some_and(|route| is_responses_endpoint(&route.endpoint));
        let is_joycode_upstream = is_joycode_provider(provider, &base_url);
        let is_joycode_anthropic_route = is_joycode_upstream
            && (devin_upstream_is_messages
                || model_catalog_route
                    .as_ref()
                    .and_then(|route| route.endpoint.as_deref())
                    .is_some_and(is_messages_endpoint)
                || (!matches!(app_type, AppType::Devin) && is_messages_endpoint(endpoint)));
        let devin_responses_codex_compat = devin_route
            .as_ref()
            .is_some_and(|route| route.responses_codex_compat);
        let devin_responses_fast_mode = devin_route
            .as_ref()
            .is_some_and(|route| route.responses_fast_mode);

        let needs_transform = match resolved_claude_api_format.as_deref() {
            Some(api_format) => super::providers::claude_api_format_needs_transform(api_format),
            None => adapter.needs_transform(provider),
        };
        let devin_local_is_messages =
            matches!(app_type, AppType::Devin) && is_messages_endpoint(endpoint);
        let devin_messages_to_chat =
            devin_local_is_messages && devin_route.is_some() && devin_upstream_is_chat;
        let devin_messages_to_responses =
            devin_local_is_messages && devin_route.is_some() && devin_upstream_is_responses;
        let devin_route_to_responses = matches!(app_type, AppType::Devin)
            && devin_route.is_some()
            && devin_upstream_is_responses
            && !value_is_openai_responses_request(&mapped_body);
        let codex_model_catalog_to_messages = model_catalog_route
            .as_ref()
            .is_some_and(|route| route.endpoint.as_deref().is_some_and(is_messages_endpoint));
        let codex_responses_to_chat = if matches!(app_type, AppType::Devin) && devin_route.is_some()
        {
            is_responses_endpoint(endpoint)
                && (devin_upstream_is_chat || devin_upstream_is_messages)
        } else if codex_model_catalog_to_messages {
            is_responses_endpoint(endpoint) || is_chat_completions_endpoint(endpoint)
        } else if model_catalog_route.as_ref().is_some_and(|route| {
            route
                .endpoint
                .as_deref()
                .is_some_and(is_chat_completions_endpoint)
        }) {
            is_responses_endpoint(endpoint)
        } else {
            matches!(app_type, AppType::Codex | AppType::Devin)
                && super::providers::should_convert_codex_responses_to_chat(provider, endpoint)
        };
        let codex_chat_to_responses = if matches!(app_type, AppType::Devin) && devin_route.is_some()
        {
            is_chat_completions_endpoint(endpoint) && devin_upstream_is_responses
        } else if model_catalog_route
            .as_ref()
            .is_some_and(|route| route.endpoint.as_deref().is_some_and(is_responses_endpoint))
        {
            is_chat_completions_endpoint(endpoint)
        } else {
            matches!(app_type, AppType::Codex | AppType::Devin)
                && super::providers::should_convert_codex_chat_to_responses(provider, endpoint)
        };
        let (effective_endpoint, passthrough_query) = if let Some(route) = devin_route.as_ref() {
            (
                replace_endpoint_path_preserve_query(endpoint, &route.endpoint),
                split_endpoint_and_query(endpoint)
                    .1
                    .map(ToString::to_string),
            )
        } else if let Some(route) = model_catalog_route.as_ref().filter(|route| {
            route
                .endpoint
                .as_deref()
                .is_some_and(|endpoint| !endpoint.is_empty())
        }) {
            (
                replace_endpoint_path_preserve_query(
                    endpoint,
                    route.endpoint.as_deref().unwrap_or(endpoint),
                ),
                split_endpoint_and_query(endpoint)
                    .1
                    .map(ToString::to_string),
            )
        } else if codex_responses_to_chat {
            rewrite_codex_responses_endpoint_to_chat(endpoint)
        } else if codex_chat_to_responses {
            rewrite_codex_chat_endpoint_to_responses(endpoint)
        } else if needs_transform && adapter.name() == "Claude" {
            let api_format = resolved_claude_api_format
                .as_deref()
                .unwrap_or_else(|| super::providers::get_claude_api_format(provider));
            rewrite_claude_transform_endpoint(endpoint, api_format, is_copilot, &mapped_body)
        } else {
            (
                endpoint.to_string(),
                split_endpoint_and_query(endpoint)
                    .1
                    .map(ToString::to_string),
            )
        };

        let codex_chat_base_is_full_endpoint = codex_responses_to_chat
            && base_url
                .trim_end_matches('/')
                .to_ascii_lowercase()
                .ends_with("/chat/completions");

        let url = if is_joycode_anthropic_route {
            build_joycode_color_gateway_url("anthropic_completions")
        } else if is_joycode_upstream && devin_upstream_is_responses {
            build_joycode_color_gateway_url("responses_completions")
        } else if is_joycode_upstream && devin_upstream_is_chat {
            build_joycode_color_gateway_url("chat_completions")
        } else if matches!(resolved_claude_api_format.as_deref(), Some("gemini_native")) {
            super::gemini_url::resolve_gemini_native_url(
                &base_url,
                &effective_endpoint,
                is_full_url,
            )
        } else if is_full_url || codex_chat_base_is_full_endpoint {
            append_query_to_full_url(&base_url, passthrough_query.as_deref())
        } else {
            adapter.build_url(&base_url, &effective_endpoint)
        };

        if is_joycode_upstream {
            let requested_model = mapped_body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("<none>");
            log::info!(
                "[JoyCode] Routing request: app={:?} provider={} requested_model={} effective_endpoint={} color_gateway={} upstream_url={}",
                app_type,
                provider.name,
                requested_model,
                effective_endpoint,
                is_joycode_anthropic_route
                    || devin_upstream_is_responses
                    || devin_upstream_is_chat,
                url
            );
        }

        // 记录映射后的出站模型名（此时 mapped_body 已完成接管映射 / [1m] 剥离 /
        // Copilot 归一化）。格式转换后若 body 仍带 model 字段会在下方刷新覆盖；
        // gemini_native 等模型在 URL 中的格式则保留此处的转换前真值。
        let mut outbound_model = mapped_body
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty())
            .map(str::to_string);

        if devin_route
            .as_ref()
            .is_some_and(|route| route.thinking_enabled == Some(false))
        {
            mapped_body = strip_devin_route_thinking(mapped_body);
        }

        // 转换请求体（如果需要）
        let mut request_body = if devin_upstream_is_messages {
            let (anthropic_body, _) = prepare_devin_anthropic_body(
                devin_request_to_anthropic_messages(mapped_body)?,
                provider,
                devin_route.as_ref(),
            );
            normalize_anthropic_temperature_for_thinking(anthropic_body)
        } else if devin_messages_to_responses || devin_route_to_responses {
            let (anthropic_body, cache_key) = prepare_devin_anthropic_body(
                devin_request_to_anthropic_messages(mapped_body)?,
                provider,
                devin_route.as_ref(),
            );
            super::providers::transform_responses::anthropic_to_responses(
                anthropic_body,
                cache_key.as_deref(),
                devin_responses_codex_compat,
                devin_responses_fast_mode,
            )?
        } else if devin_messages_to_chat {
            let (anthropic_body, _) = prepare_devin_anthropic_body(
                devin_request_to_anthropic_messages(mapped_body)?,
                provider,
                devin_route.as_ref(),
            );
            let mut chat_body = super::providers::transform::anthropic_to_openai(anthropic_body)?;
            super::providers::transform::inject_openai_stream_include_usage(&mut chat_body);
            if !is_joycode_upstream {
                inject_devin_chat_cache_key(&mut chat_body, provider, devin_route.as_ref());
            }
            chat_body
        } else if matches!(app_type, AppType::Devin)
            && codex_chat_to_responses
            && devin_route.is_some()
        {
            let (anthropic_body, cache_key) = prepare_devin_anthropic_body(
                devin_request_to_anthropic_messages(mapped_body)?,
                provider,
                devin_route.as_ref(),
            );
            super::providers::transform_responses::anthropic_to_responses(
                anthropic_body,
                cache_key.as_deref(),
                devin_responses_codex_compat,
                devin_responses_fast_mode,
            )?
        } else if matches!(app_type, AppType::Devin)
            && devin_upstream_is_responses
            && devin_responses_codex_compat
            && is_responses_endpoint(endpoint)
        {
            let mut body =
                apply_devin_codex_responses_compat(mapped_body, devin_responses_fast_mode);
            inject_devin_responses_cache_key(&mut body, provider, devin_route.as_ref());
            body
        } else if codex_model_catalog_to_messages {
            codex_request_to_anthropic_messages(mapped_body)?
        } else if codex_responses_to_chat {
            let mut mapped_body = mapped_body;
            let restored = self
                .codex_chat_history
                .enrich_request(&mut mapped_body)
                .await;
            if restored > 0 {
                log::debug!(
                    "[Codex] Restored or enriched {restored} cached function call item(s) for Chat upstream"
                );
            }
            super::providers::apply_codex_chat_upstream_model(provider, &mut mapped_body);
            let reasoning_config =
                super::providers::resolve_codex_chat_reasoning_config(provider, &mapped_body);
            super::providers::transform_codex_chat::responses_to_chat_completions_with_reasoning(
                mapped_body,
                reasoning_config.as_ref(),
            )?
        } else if codex_chat_to_responses {
            super::providers::transform_codex_chat::chat_completions_to_responses(mapped_body)?
        } else if matches!(app_type, AppType::Devin) && value_is_openai_chat_request(&mapped_body) {
            compact_devin_openai_chat_context(mapped_body)
        } else if needs_transform {
            if adapter.name() == "Claude" {
                let api_format = resolved_claude_api_format
                    .as_deref()
                    .unwrap_or_else(|| super::providers::get_claude_api_format(provider));
                super::providers::transform_claude_request_for_api_format(
                    mapped_body,
                    provider,
                    api_format,
                    self.session_client_provided
                        .then_some(self.session_id.as_str()),
                    Some(self.gemini_shadow.as_ref()),
                )?
            } else {
                adapter.transform_request(mapped_body, provider)?
            }
        } else {
            mapped_body
        };

        if matches!(app_type, AppType::Codex | AppType::Devin) {
            self.apply_media_prevention(&mut request_body, provider);
        }

        // 过滤私有参数（以 `_` 开头的字段），防止内部信息泄露到上游
        // 默认使用空白名单，过滤所有 _ 前缀字段
        let mut filtered_body = prepare_upstream_request_body(request_body);
        if !is_copilot {
            if let Some(overrides) = provider
                .meta
                .as_ref()
                .and_then(|meta| meta.local_proxy_request_overrides.as_ref())
            {
                if apply_local_proxy_body_overrides(&mut filtered_body, overrides) {
                    filtered_body = prepare_upstream_request_body(filtered_body);
                }
            }
        }
        let sensitive_rewrite_map = if matches!(app_type, AppType::Devin) {
            super::sensitive_redaction::pseudonymize_sensitive_value(&mut filtered_body)
        } else {
            super::sensitive_redaction::SensitiveRewriteMap::default()
        };
        // 出站 body 定稿后刷新真值（覆盖 Codex chat 上游模型覆写、转换层模型改写）
        if let Some(m) = filtered_body
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty())
        {
            outbound_model = Some(m.to_string());
        }
        log_prompt_cache_trace(
            app_type,
            provider,
            &effective_endpoint,
            resolved_claude_api_format.as_deref(),
            &filtered_body,
            self.session_client_provided,
        );
        if matches!(app_type, AppType::Devin) {
            log_devin_chat_prefix_trace(provider, &effective_endpoint, &filtered_body);
        }
        let request_is_streaming =
            is_streaming_request(&effective_endpoint, &filtered_body, headers);
        let force_identity_encoding = needs_transform
            || devin_messages_to_chat
            || devin_messages_to_responses
            || codex_responses_to_chat
            || codex_chat_to_responses
            || is_joycode_upstream
            || (request_is_streaming && !is_joycode_anthropic_route);

        // Codex OAuth 需要注入的 ChatGPT-Account-Id（在动态 token 获取期间填充）
        let mut codex_oauth_account_id: Option<String> = None;
        let mut should_send_codex_oauth_session_headers = false;

        // 获取认证头（提前准备，用于内联替换）
        let mut auth_headers = if is_joycode_anthropic_route {
            Vec::new()
        } else if let Some(route) = devin_route.as_ref().filter(|route| {
            route
                .api_key
                .as_deref()
                .map(str::trim)
                .is_some_and(|key| !key.is_empty())
        }) {
            devin_route_auth_headers(route)?
        } else if let Some(route) = model_catalog_route.as_ref().filter(|route| {
            route
                .api_key
                .as_deref()
                .map(str::trim)
                .is_some_and(|key| !key.is_empty())
        }) {
            model_catalog_route_auth_headers(route)?
        } else if let Some(mut auth) = adapter.extract_auth(provider) {
            // GitHub Copilot 特殊处理：从 CopilotAuthManager 获取真实 token
            if auth.strategy == AuthStrategy::GitHubCopilot {
                if let Some(app_handle) = &self.app_handle {
                    let copilot_state = app_handle.state::<CopilotAuthState>();
                    let copilot_auth: tokio::sync::RwLockReadGuard<'_, CopilotAuthManager> =
                        copilot_state.0.read().await;

                    // 从 provider.meta 获取关联的 GitHub 账号 ID（多账号支持）
                    let account_id = provider
                        .meta
                        .as_ref()
                        .and_then(|m| m.managed_account_id_for("github_copilot"));

                    // 根据账号 ID 获取对应 token（向后兼容：无账号 ID 时使用第一个账号）
                    let token_result = match &account_id {
                        Some(id) => {
                            log::debug!("[Copilot] 使用指定账号 {id} 获取 token");
                            copilot_auth.get_valid_token_for_account(id).await
                        }
                        None => {
                            log::debug!("[Copilot] 使用默认账号获取 token");
                            copilot_auth.get_valid_token().await
                        }
                    };

                    match token_result {
                        Ok(token) => {
                            auth = AuthInfo::new(token, AuthStrategy::GitHubCopilot);
                            log::debug!(
                                "[Copilot] 成功获取 Copilot token (account={})",
                                account_id.as_deref().unwrap_or("default")
                            );
                        }
                        Err(e) => {
                            log::error!(
                                "[Copilot] 获取 Copilot token 失败 (account={}): {e}",
                                account_id.as_deref().unwrap_or("default")
                            );
                            return Err(ProxyError::AuthError(format!(
                                "GitHub Copilot 认证失败: {e}"
                            )));
                        }
                    }
                } else {
                    log::error!("[Copilot] AppHandle 不可用");
                    return Err(ProxyError::AuthError(
                        "GitHub Copilot 认证不可用（无 AppHandle）".to_string(),
                    ));
                }
            }

            // Codex OAuth 特殊处理：从 CodexOAuthManager 获取真实 access_token
            if auth.strategy == AuthStrategy::CodexOAuth {
                if let Some(app_handle) = &self.app_handle {
                    let codex_state = app_handle.state::<CodexOAuthState>();
                    let codex_auth: tokio::sync::RwLockReadGuard<'_, CodexOAuthManager> =
                        codex_state.0.read().await;

                    // 从 provider.meta 获取关联的 ChatGPT 账号 ID
                    let account_id = provider
                        .meta
                        .as_ref()
                        .and_then(|m| m.managed_account_id_for("codex_oauth"));

                    let token_result = match &account_id {
                        Some(id) => {
                            log::debug!("[CodexOAuth] 使用指定账号 {id} 获取 token");
                            codex_auth.get_valid_token_for_account(id).await
                        }
                        None => {
                            log::debug!("[CodexOAuth] 使用默认账号获取 token");
                            codex_auth.get_valid_token().await
                        }
                    };

                    match token_result {
                        Ok(token) => {
                            auth = AuthInfo::new(token, AuthStrategy::CodexOAuth);
                            should_send_codex_oauth_session_headers = true;
                            // 解析使用的 account_id（用于注入 ChatGPT-Account-Id header）
                            codex_oauth_account_id = match account_id {
                                Some(id) => Some(id),
                                None => codex_auth.default_account_id().await,
                            };
                            log::debug!(
                                "[CodexOAuth] 成功获取 access_token (account={})",
                                codex_oauth_account_id.as_deref().unwrap_or("default")
                            );
                        }
                        Err(e) => {
                            log::error!("[CodexOAuth] 获取 access_token 失败: {e}");
                            return Err(ProxyError::AuthError(format!(
                                "Codex OAuth 认证失败: {e}"
                            )));
                        }
                    }
                } else {
                    log::error!("[CodexOAuth] AppHandle 不可用");
                    return Err(ProxyError::AuthError(
                        "Codex OAuth 认证不可用（无 AppHandle）".to_string(),
                    ));
                }
            }

            adapter.get_auth_headers(&auth)?
        } else {
            Vec::new()
        };

        // 注入 Codex OAuth 的 ChatGPT-Account-Id header（如果有 account_id）
        if let Some(ref account_id) = codex_oauth_account_id {
            if let Ok(hv) = http::HeaderValue::from_str(account_id) {
                auth_headers.push((http::HeaderName::from_static("chatgpt-account-id"), hv));
            }
        }

        let codex_oauth_session_headers =
            if should_send_codex_oauth_session_headers && self.session_client_provided {
                build_codex_oauth_session_headers(&self.session_id)
            } else {
                Vec::new()
            };

        // 自定义 User-Agent：与 stream_check / model_fetch 共用 parse_custom_user_agent，
        // 运行时静默忽略非法值（前端在输入处给非阻断提示，不在保存时阻断）。
        // Copilot 指纹 UA 不可覆盖。
        let custom_user_agent = if is_copilot {
            None
        } else {
            provider
                .meta
                .as_ref()
                .and_then(|meta| meta.custom_user_agent_header().ok().flatten())
        };
        let devin_extra_headers = devin_route
            .as_ref()
            .map(|route| route.extra_headers.clone())
            .unwrap_or_default();
        if matches!(app_type, AppType::Devin) && !devin_extra_headers.is_empty() {
            log::debug!(
                "[Devin] Extra headers from route: {} headers",
                devin_extra_headers.len()
            );
            for (name, value) in &devin_extra_headers {
                log::debug!(
                    "[Devin] Route header: {}: {}",
                    name,
                    redact_header_value_for_log(name, value)
                );
            }
        }
        let devin_default_user_agent =
            if matches!(app_type, AppType::Devin) && devin_route.is_some() {
                if is_joycode_anthropic_route {
                    None
                } else if devin_upstream_is_messages {
                    Some(http::HeaderValue::from_static(
                        "claude-cli/2.1.2 (external, cli)",
                    ))
                } else {
                    Some(http::HeaderValue::from_static("codex_cli_rs/0.80.0"))
                }
            } else {
                None
            };

        // --- Copilot 优化器：动态 header 注入 ---
        if let Some((ref classification, ref det_request_id, ref interaction_id)) =
            copilot_optimization
        {
            for (name, value) in auth_headers.iter_mut() {
                match name.as_str() {
                    "x-initiator" if self.copilot_optimizer_config.request_classification => {
                        *value = http::HeaderValue::from_static(classification.initiator);
                    }
                    "x-interaction-type" if classification.is_subagent => {
                        // 子代理请求：conversation-subagent 不计 premium interaction
                        *value = http::HeaderValue::from_static("conversation-subagent");
                    }
                    "x-request-id" | "x-agent-task-id" => {
                        if let Some(ref det_id) = det_request_id {
                            if let Ok(hv) = http::HeaderValue::from_str(det_id) {
                                *value = hv;
                            }
                        }
                    }
                    _ => {}
                }
            }

            // x-interaction-id：仅在有 session 时注入（不在 get_auth_headers 中）
            if let Some(ref iid) = interaction_id {
                if let Ok(hv) = http::HeaderValue::from_str(iid) {
                    auth_headers.push((http::HeaderName::from_static("x-interaction-id"), hv));
                }
            }

            if classification.is_subagent {
                log::info!(
                    "[Copilot] 子代理请求: x-initiator=agent, x-interaction-type=conversation-subagent"
                );
            }
        }

        // Copilot 指纹头名（由 get_auth_headers 注入，需在原始头中去重）
        let copilot_fingerprint_headers: &[&str] = if is_copilot {
            &[
                "user-agent",
                "editor-version",
                "editor-plugin-version",
                "copilot-integration-id",
                "x-github-api-version",
                "openai-intent",
                // 新增 headers
                "x-initiator",
                "x-interaction-type",
                "x-interaction-id",
                "x-vscode-user-agent-library-version",
                "x-request-id",
                "x-agent-task-id",
            ]
        } else {
            &[]
        };

        // 预计算上游 host 值（用于在原位替换 host header）
        let upstream_host = url
            .parse::<http::Uri>()
            .ok()
            .and_then(|u| u.authority().map(|a| a.to_string()));

        let should_send_anthropic_headers = !is_joycode_anthropic_route
            && ((adapter.name() == "Claude"
                && matches!(resolved_claude_api_format.as_deref(), Some("anthropic")))
                || devin_upstream_is_messages
                || codex_model_catalog_to_messages);
        let rebuild_json_headers_for_json_upstream =
            should_rebuild_connect_headers_for_json_upstream(
                app_type,
                devin_route.is_some(),
                devin_messages_to_chat,
                devin_messages_to_responses,
                devin_route_to_responses,
                devin_upstream_is_responses,
                devin_upstream_is_messages,
                codex_model_catalog_to_messages,
            );

        // 预计算 anthropic-beta 值（仅 Claude）
        let anthropic_beta_value = if should_send_anthropic_headers {
            const CLAUDE_CODE_BETA: &str = "claude-code-20250219";
            let mut beta_parts = vec![CLAUDE_CODE_BETA.to_string()];
            if let Some(beta) = headers.get("anthropic-beta") {
                if let Ok(beta_str) = beta.to_str() {
                    append_anthropic_beta_tokens(&mut beta_parts, beta_str);
                }
            }
            for (_, value) in devin_extra_headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            {
                append_anthropic_beta_tokens(&mut beta_parts, value);
            }
            Some(beta_parts.join(","))
        } else {
            None
        };

        // ============================================================
        // JoyCode Anthropic 特殊处理：调用 prepare 获取 X-Model-Token
        // ============================================================
        let mut joycode_model_token: Option<String> = None;
        let mut joycode_chat_id: Option<String> = None;
        let joycode_provider_headers = if is_joycode_upstream {
            collect_joycode_provider_headers(provider)
        } else {
            Vec::new()
        };
        let joycode_auth_params = if is_joycode_upstream {
            let mut headers = joycode_provider_headers.clone();
            headers.extend(devin_extra_headers.clone());
            extract_joycode_auth_params(&headers)
        } else {
            None
        };
        if is_joycode_upstream {
            log::info!(
                "[JoyCode] Auth state before upstream call: {}",
                if joycode_auth_params.is_some() {
                    "present"
                } else {
                    "missing"
                }
            );
        }
        if is_joycode_upstream {
            let prepare_base_url = devin_route
                .as_ref()
                .and_then(|route| route.base_url.as_deref())
                .unwrap_or(&base_url);
            if is_joycode_anthropic_route {
                log::debug!("[JoyCode] Detected Anthropic endpoint, calling prepare API");
            } else {
                log::debug!("[JoyCode] Detected JoyCode endpoint, calling prepare API");
            }
            log::debug!(
                "[JoyCode] Available extra_headers count: {}",
                devin_extra_headers.len()
            );
            for (k, v) in &devin_extra_headers {
                log::debug!(
                    "[JoyCode]   header: {} = {}",
                    k,
                    if k.eq_ignore_ascii_case("cookie")
                        || k.eq_ignore_ascii_case("ptKey")
                        || k.eq_ignore_ascii_case("ptkey")
                        || k.eq_ignore_ascii_case("pt_key")
                        || k.eq_ignore_ascii_case("x-pt-key")
                    {
                        format!("<redacted,len={}>", v.len())
                    } else {
                        v.clone()
                    }
                );
            }

            if let Some((pt_key, login_type, tenant)) = joycode_auth_params.as_ref() {
                log::debug!(
                    "[JoyCode] Extracted auth: pt_key=<redacted,len={}>, loginType={}, tenant={}",
                    pt_key.len(),
                    login_type,
                    tenant
                );

                // 调用 JoyCode prepare API
                let adapter = JoyCodeAnthropicAdapter::new();
                let upstream_model = outbound_model
                    .as_deref()
                    .filter(|model| !model.is_empty())
                    .unwrap_or("Claude-Sonnet-4.6-hq");

                match adapter
                    .get_model_prepare(
                        prepare_base_url,
                        upstream_model,
                        request_is_streaming,
                        pt_key,
                        login_type,
                        tenant,
                        "京东集团", // orgFullName
                    )
                    .await
                {
                    Ok(prepared) => {
                        log::info!("[JoyCode] X-Model-Token obtained for {}", upstream_model);
                        joycode_chat_id = Some(prepared.chat_id);
                        joycode_model_token = Some(prepared.token);
                    }
                    Err(e) => {
                        log::error!("[JoyCode] Failed to get model token: {}", e);
                        return Err(e);
                    }
                }
            } else {
                log::warn!("[JoyCode] Missing ptKey for Anthropic prepare");
            }
        }

        // ============================================================
        // 构建有序 HeaderMap — 内联替换，保持客户端原始顺序
        // ============================================================
        let mut ordered_headers = http::HeaderMap::new();
        let mut saw_auth = false;
        let mut saw_accept_encoding = false;
        let mut saw_user_agent = false;
        let mut saw_anthropic_beta = false;
        let mut saw_anthropic_version = false;

        for (key, value) in headers {
            let key_str = key.as_str();

            // --- host — 原位替换为上游 host（保持客户端原始位置） ---
            if key_str.eq_ignore_ascii_case("host") {
                if let Some(ref host_val) = upstream_host {
                    if let Ok(hv) = http::HeaderValue::from_str(host_val) {
                        ordered_headers.append(key.clone(), hv);
                    }
                }
                continue;
            }

            // Devin/Windsurf/Codex 本地入口可能是 Connect-RPC/protobuf。
            // 上游已经被转换成 JSON API，不能把原 content-type/content-encoding
            // 或 connect/grpc 私有头继续带过去，否则 NewAPI 会按非 JSON 解析 body。
            if rebuild_json_headers_for_json_upstream
                && is_connect_header_for_json_upstream(key_str)
            {
                continue;
            }

            // Match JoyCode's official VS Code extension for model calls.
            // Extra Accept/User-Agent/tenant headers trigger the upgrade gate.
            if is_joycode_upstream
                && (key_str.eq_ignore_ascii_case("accept")
                    || key_str.eq_ignore_ascii_case("user-agent")
                    || key_str.eq_ignore_ascii_case("tenant"))
            {
                continue;
            }

            // --- 连接 / 追踪 / CDN 类 — 无条件跳过 ---
            if matches!(
                key_str,
                "content-length"
                    | "transfer-encoding"
                    | "x-forwarded-host"
                    | "x-forwarded-port"
                    | "x-forwarded-proto"
                    | "forwarded"
                    | "cf-connecting-ip"
                    | "cf-ipcountry"
                    | "cf-ray"
                    | "cf-visitor"
                    | "true-client-ip"
                    | "fastly-client-ip"
                    | "x-azure-clientip"
                    | "x-azure-fdid"
                    | "x-azure-ref"
                    | "akamai-origin-hop"
                    | "x-akamai-config-log-detail"
                    | "x-request-id"
                    | "x-correlation-id"
                    | "x-trace-id"
                    | "x-amzn-trace-id"
                    | "x-b3-traceid"
                    | "x-b3-spanid"
                    | "x-b3-parentspanid"
                    | "x-b3-sampled"
                    | "traceparent"
                    | "tracestate"
            ) {
                continue;
            }

            // --- 认证类 — 用 adapter 提供的认证头替换（在原始位置） ---
            if key_str.eq_ignore_ascii_case("authorization")
                || key_str.eq_ignore_ascii_case("x-api-key")
                || key_str.eq_ignore_ascii_case("x-goog-api-key")
            {
                if !saw_auth {
                    saw_auth = true;
                    for (ah_name, ah_value) in &auth_headers {
                        ordered_headers.append(ah_name.clone(), ah_value.clone());
                    }

                    // JoyCode Anthropic: 注入额外的认证头
                    if let Some(ref token) = joycode_model_token {
                        if let Ok(token_value) = http::HeaderValue::from_str(token) {
                            ordered_headers.append(
                                http::HeaderName::from_static("x-model-token"),
                                token_value,
                            );
                        }

                        // 注入 ptKey, loginType, tenant
                        for (name, value) in &devin_extra_headers {
                            if name.eq_ignore_ascii_case("ptKey")
                                || name.eq_ignore_ascii_case("loginType")
                                || name.eq_ignore_ascii_case("tenant")
                            {
                                if let Ok(hv) = http::HeaderValue::from_str(value) {
                                    if let Ok(hn) = http::HeaderName::from_bytes(name.as_bytes()) {
                                        ordered_headers.append(hn, hv);
                                    }
                                }
                            }
                        }

                        log::debug!("[JoyCode] Injected X-Model-Token and auth headers");
                    }
                }
                continue;
            }

            // --- accept-encoding — transform / SSE 路径强制 identity，其余保留原值 ---
            if key_str.eq_ignore_ascii_case("accept-encoding") {
                if !saw_accept_encoding {
                    saw_accept_encoding = true;
                    if force_identity_encoding {
                        ordered_headers.append(
                            http::header::ACCEPT_ENCODING,
                            http::HeaderValue::from_static("identity"),
                        );
                    } else {
                        ordered_headers.append(key.clone(), value.clone());
                    }
                }
                continue;
            }

            // --- user-agent: provider-level override for local proxy routing ---
            if !is_copilot && key_str.eq_ignore_ascii_case("user-agent") {
                if !saw_user_agent {
                    saw_user_agent = true;
                    if let Some(ref ua) = custom_user_agent {
                        ordered_headers.append(http::header::USER_AGENT, ua.clone());
                    } else if let Some(ref ua) = devin_default_user_agent {
                        ordered_headers.append(http::header::USER_AGENT, ua.clone());
                    } else {
                        ordered_headers.append(key.clone(), value.clone());
                    }
                }
                continue;
            }

            // --- anthropic-beta — 用重建值替换（确保含 claude-code 标记） ---
            if key_str.eq_ignore_ascii_case("anthropic-beta") {
                if !saw_anthropic_beta {
                    saw_anthropic_beta = true;
                    if let Some(ref beta_val) = anthropic_beta_value {
                        if let Ok(hv) = http::HeaderValue::from_str(beta_val) {
                            ordered_headers.append("anthropic-beta", hv);
                        }
                    }
                }
                continue;
            }

            // --- anthropic-version — 透传客户端值 ---
            if key_str.eq_ignore_ascii_case("anthropic-version") {
                if should_send_anthropic_headers {
                    saw_anthropic_version = true;
                    ordered_headers.append(key.clone(), value.clone());
                }
                continue;
            }

            // --- Copilot 指纹头 — 跳过（由 auth_headers 提供） ---
            if copilot_fingerprint_headers
                .iter()
                .any(|h| key_str.eq_ignore_ascii_case(h))
            {
                continue;
            }

            // --- 默认：透传 ---
            ordered_headers.append(key.clone(), value.clone());
        }

        // 如果原始请求中没有认证头，在末尾追加
        if !saw_auth && !auth_headers.is_empty() {
            let app_type_str = match app_type {
                AppType::Claude => "Claude",
                AppType::ClaudeDesktop => "ClaudeDesktop",
                AppType::Codex => "Codex",
                AppType::Gemini => "Gemini",
                AppType::OpenCode => "OpenCode",
                AppType::OpenClaw => "OpenClaw",
                AppType::Hermes => "Hermes",
                AppType::Devin => "Devin",
            };
            log::debug!(
                "[{}] Injecting {} auth headers",
                app_type_str,
                auth_headers.len()
            );
            for (ah_name, ah_value) in &auth_headers {
                log::debug!(
                    "[{}] Auth header: {}: {}",
                    app_type_str,
                    ah_name,
                    redact_header_value_for_log(
                        ah_name.as_str(),
                        ah_value.to_str().unwrap_or("<binary>")
                    )
                );
                ordered_headers.append(ah_name.clone(), ah_value.clone());
            }
        }

        // transform / SSE 路径在缺失时补 identity；普通透传不主动补 accept-encoding
        if !saw_accept_encoding && force_identity_encoding {
            ordered_headers.append(
                http::header::ACCEPT_ENCODING,
                http::HeaderValue::from_static("identity"),
            );
        }

        if !saw_user_agent {
            if let Some(ref ua) = custom_user_agent {
                ordered_headers.append(http::header::USER_AGENT, ua.clone());
            } else if let Some(ref ua) = devin_default_user_agent {
                ordered_headers.append(http::header::USER_AGENT, ua.clone());
            }
        }

        // 如果原始请求中没有 anthropic-beta 且有值需要添加，追加
        if !saw_anthropic_beta {
            if let Some(ref beta_val) = anthropic_beta_value {
                if let Ok(hv) = http::HeaderValue::from_str(beta_val) {
                    ordered_headers.append("anthropic-beta", hv);
                }
            }
        }

        // anthropic-version：仅在缺失时补充默认值
        if should_send_anthropic_headers && !saw_anthropic_version {
            ordered_headers.append(
                "anthropic-version",
                http::HeaderValue::from_static("2023-06-01"),
            );
        }

        // JoyCode Anthropic: 注入 X-Model-Token 和认证头
        if let Some(ref token) = joycode_model_token {
            log::debug!("[JoyCode] Injecting headers, token length: {}", token.len());

            // X-Model-Token
            if let Ok(token_value) = http::HeaderValue::from_str(token) {
                ordered_headers.insert(http::HeaderName::from_static("x-model-token"), token_value);
                log::debug!("[JoyCode] Injected x-model-token");
            }
            if !ordered_headers.contains_key("x-ms-client-request-id") {
                let request_id = uuid::Uuid::new_v4().to_string();
                if let Ok(value) = http::HeaderValue::from_str(&request_id) {
                    ordered_headers.insert(
                        http::HeaderName::from_static("x-ms-client-request-id"),
                        value,
                    );
                }
            }

            // 注入 ptKey, loginType, tenant（使用保存的认证参数）
            let injected_count =
                append_joycode_auth_headers(&mut ordered_headers, joycode_auth_params.as_ref());
            if is_joycode_upstream {
                ordered_headers.remove("tenant");
            }

            log::info!(
                "[JoyCode] Injected {} auth headers (x-model-token + {} from auth_params)",
                1 + injected_count,
                injected_count
            );
        } else if is_joycode_upstream {
            let injected_count =
                append_joycode_auth_headers(&mut ordered_headers, joycode_auth_params.as_ref());
            ordered_headers.remove("tenant");
            if injected_count > 0 {
                log::info!(
                    "[JoyCode] Injected {} auth headers for OpenAI-compatible route",
                    injected_count
                );
            }
        }

        if is_joycode_anthropic_route {
            if let Some(chat_id) = joycode_chat_id.as_deref() {
                if let Some(object) = filtered_body.as_object_mut() {
                    object.insert(
                        "chatId".to_string(),
                        serde_json::Value::String(chat_id.to_string()),
                    );
                    object.insert(
                        "tenant".to_string(),
                        serde_json::Value::String(
                            joycode_auth_params
                                .as_ref()
                                .map(|(_, _, tenant)| tenant.clone())
                                .unwrap_or_else(|| "JD".to_string()),
                        ),
                    );
                    object
                        .entry("orgFullName".to_string())
                        .or_insert_with(|| serde_json::Value::String("京东集团".to_string()));
                    object
                        .entry("userId".to_string())
                        .or_insert_with(|| serde_json::Value::String(String::new()));
                    object.insert(
                        "client".to_string(),
                        serde_json::Value::String(JOYCODE_VSCODE_CLIENT.to_string()),
                    );
                    object.insert(
                        "clientVersion".to_string(),
                        serde_json::Value::String(JOYCODE_VSCODE_CLIENT_VERSION.to_string()),
                    );
                    object
                        .entry("language".to_string())
                        .or_insert_with(|| serde_json::Value::String("UNKNOWN".to_string()));
                    log::debug!("[JoyCode] Injected chatId into request body");
                }
            }
        }

        // Codex OAuth 反代尽量对齐官方 Codex CLI 的会话路由信号。
        // 只发送客户端提供的 session_id；生成的 UUID 每次不同，反而会破坏前缀缓存。
        for (name, value) in codex_oauth_session_headers {
            ordered_headers.insert(name, value);
        }

        // 序列化请求体。GET/HEAD 是 idempotent/safe 方法，按 HTTP 语义不应携带 body；
        // 强行附带 JSON body 会让某些上游（如 Google Gemini 的 models.list）拒绝请求。
        let body_bytes = if matches!(method, &http::Method::GET | &http::Method::HEAD) {
            Vec::new()
        } else {
            serde_json::to_vec(&filtered_body).map_err(|e| {
                ProxyError::Internal(format!("Failed to serialize request body: {e}"))
            })?
        };

        if rebuild_json_headers_for_json_upstream {
            ordered_headers.insert(
                http::header::CONTENT_TYPE,
                if is_joycode_upstream {
                    http::HeaderValue::from_static("application/json; charset=UTF-8")
                } else {
                    http::HeaderValue::from_static("application/json")
                },
            );
            if !is_joycode_upstream {
                ordered_headers.insert(
                    http::header::ACCEPT,
                    http::HeaderValue::from_static("text/event-stream"),
                );
            }
        } else if !ordered_headers.contains_key(http::header::CONTENT_TYPE) {
            ordered_headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("application/json"),
            );
        }

        for (name, value) in &devin_extra_headers {
            if is_joycode_anthropic_route || name.eq_ignore_ascii_case("anthropic-beta") {
                continue;
            }
            let Ok(header_name) = http::HeaderName::from_bytes(name.as_bytes()) else {
                log::warn!("[Devin] Ignoring invalid route header name: {name}");
                continue;
            };
            let Ok(header_value) = http::HeaderValue::from_str(value) else {
                log::warn!("[Devin] Ignoring invalid route header value for: {name}");
                continue;
            };
            ordered_headers.insert(header_name, header_value);
        }

        apply_local_proxy_header_overrides(
            &mut ordered_headers,
            provider
                .meta
                .as_ref()
                .and_then(|meta| meta.local_proxy_request_overrides.as_ref()),
            is_copilot,
        );

        reject_proxy_placeholder_for_managed_account_upstream(&url, &ordered_headers)?;

        // 输出请求信息日志
        let tag = adapter.name();
        let request_model = filtered_body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("<none>");
        log::info!("[{tag}] >>> 请求 URL: {url} (model={request_model})");
        if log::log_enabled!(log::Level::Debug) {
            if let Ok(body_str) = serde_json::to_string(&filtered_body) {
                log::debug!(
                    "[{tag}] >>> 请求体内容 ({}字节): {}",
                    body_str.len(),
                    body_str
                );
            }
        }

        // 确定超时
        let timeout = if self.non_streaming_timeout.is_zero() {
            std::time::Duration::from_secs(600) // 默认 600 秒
        } else {
            self.non_streaming_timeout
        };

        // 获取全局代理 URL
        let upstream_proxy_url: Option<String> = super::http_client::get_current_proxy_url();

        // SOCKS5 代理不支持 CONNECT 隧道，需要用 reqwest
        let is_socks_proxy = upstream_proxy_url
            .as_deref()
            .map(|u| u.starts_with("socks5"))
            .unwrap_or(false);

        let preserve_exact_header_case = should_preserve_exact_header_case(
            adapter.name(),
            provider,
            resolved_claude_api_format.as_deref(),
            is_copilot,
        );

        // 发送请求
        let response = if is_socks_proxy || !preserve_exact_header_case {
            // OpenAI / Copilot / Codex 类后端不依赖原始 header 大小写；走 reqwest
            // 连接池，避免 raw TCP/TLS path 每次请求都重新握手。SOCKS5 也只能走 reqwest。
            log::debug!(
                "[Forwarder] Using pooled reqwest client (preserve_exact_header_case={preserve_exact_header_case}, socks_proxy={is_socks_proxy})"
            );

            // 调试：打印最终要发送的所有头
            if matches!(app_type, AppType::Devin) && log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "[Devin] Final headers to send ({} total):",
                    ordered_headers.len()
                );
                for (key, value) in &ordered_headers {
                    log::debug!(
                        "[Devin]   {}: {}",
                        key,
                        redact_header_value_for_log(
                            key.as_str(),
                            value.to_str().unwrap_or("<binary>")
                        )
                    );
                }
            }

            let use_direct_client =
                should_use_direct_http_client_for_upstream(provider, devin_route.as_ref());
            let client = if use_direct_client {
                log::debug!(
                    "[Forwarder] Using direct HTTP client for upstream request (bypass system proxy): provider={}, url={}",
                    provider.name,
                    url
                );
                super::http_client::get_direct()
            } else {
                super::http_client::get()
            };
            let mut request = client.request(method.clone(), &url);
            if request_is_streaming {
                // reqwest 的 timeout 是整请求超时；流式请求交给 response_processor
                // 的首包/静默期超时控制，避免长流被总时长误杀。
                request = request.timeout(std::time::Duration::from_secs(24 * 60 * 60));
            } else if !self.non_streaming_timeout.is_zero() {
                request = request.timeout(self.non_streaming_timeout);
            }
            for (key, value) in &ordered_headers {
                request = request.header(key, value);
            }
            let outbound_body_len = body_bytes.len();
            let send = request.body(body_bytes).send();
            let send_result = if request_is_streaming {
                let header_timeout = if self.streaming_first_byte_timeout.is_zero() {
                    timeout
                } else {
                    self.streaming_first_byte_timeout
                };
                tokio::time::timeout(header_timeout, send)
                    .await
                    .map_err(|_| {
                        ProxyError::Timeout(format!(
                            "流式响应首包超时: {}s（上游未返回响应头）",
                            header_timeout.as_secs()
                        ))
                    })?
            } else {
                send.await
            };
            let reqwest_resp = send_result.map_err(|error| {
                log_reqwest_send_error(
                    &error,
                    app_type.as_str(),
                    &provider.name,
                    &url,
                    outbound_body_len,
                );
                map_reqwest_send_error(error)
            })?;
            ProxyResponse::Reqwest(reqwest_resp)
        } else {
            // HTTP 代理或直连：走 hyper raw write（保持 header 大小写）
            // 如果有 HTTP 代理，hyper_client 会用 CONNECT 隧道穿过代理
            let uri: http::Uri = url
                .parse()
                .map_err(|e| ProxyError::ForwardFailed(format!("Invalid URL '{url}': {e}")))?;
            super::hyper_client::send_request(
                uri,
                method.clone(),
                ordered_headers,
                extensions.clone(),
                body_bytes,
                timeout,
                upstream_proxy_url.as_deref(),
            )
            .await?
        };

        // 检查响应状态
        let status = response.status();

        if status.is_success() {
            let mut response = self
                .prepare_success_response_for_failover(response, request_is_streaming)
                .await?;
            if !sensitive_rewrite_map.is_empty() {
                response =
                    restore_sensitive_pseudonyms_in_response(response, sensitive_rewrite_map)
                        .await?;
            }
            Ok((
                response,
                resolved_claude_api_format,
                outbound_model,
                codex_responses_to_chat,
                codex_chat_to_responses,
            ))
        } else {
            let status_code = status.as_u16();
            // 错误响应同样可能被上游压缩（content-encoding）。reqwest 未启用任何
            // 自动解压 feature，这里拿到的是原始字节；不解压的话，压缩过的错误体会
            // 在 from_utf8 处变成非 UTF-8 而被丢弃，隐藏掉上游的限流/鉴权等详情。
            let encoding = get_content_encoding(response.headers());
            let raw = response.bytes().await?;
            let decoded = match encoding {
                Some(encoding) => match decompress_body(&encoding, &raw) {
                    Ok(Some(decompressed)) => decompressed,
                    // 不支持的编码 / 解压失败：退回原始字节，尽量保留可读信息
                    _ => raw.to_vec(),
                },
                None => raw.to_vec(),
            };
            let body_text = String::from_utf8(decoded).ok();

            Err(ProxyError::UpstreamError {
                status: status_code,
                body: body_text,
            })
        }
    }

    /// 故障转移开启时，成功不能只看上游响应头。
    ///
    /// - 非流式：先把完整 body 读到内存，读超时/连接中断会回到 retry loop 尝试下一家。
    /// - 流式：至少等首个 chunk 到达，避免上游返回 200 后一直不吐 SSE 时被误记成功。
    async fn prepare_success_response_for_failover(
        &self,
        response: ProxyResponse,
        request_is_streaming: bool,
    ) -> Result<ProxyResponse, ProxyError> {
        if request_is_streaming {
            return self.prime_streaming_response(response).await;
        }

        if self.non_streaming_timeout.is_zero() {
            return Ok(response);
        }

        let status = response.status();
        let headers = response.headers().clone();
        let body_timeout = self.non_streaming_timeout;
        let body = tokio::time::timeout(body_timeout, response.bytes())
            .await
            .map_err(|_| {
                ProxyError::Timeout(format!(
                    "响应体读取超时: {}s（上游发完响应头后 body 未到达）",
                    body_timeout.as_secs()
                ))
            })??;

        Ok(ProxyResponse::buffered(status, headers, body))
    }

    async fn prime_streaming_response(
        &self,
        response: ProxyResponse,
    ) -> Result<ProxyResponse, ProxyError> {
        if self.streaming_first_byte_timeout.is_zero() {
            return Ok(response);
        }

        let status = response.status();
        let headers = response.headers().clone();
        let timeout = self.streaming_first_byte_timeout;
        let mut stream = Box::pin(response.bytes_stream());

        let first = tokio::time::timeout(timeout, stream.next())
            .await
            .map_err(|_| {
                ProxyError::Timeout(format!(
                    "流式响应首包超时: {}s（上游已返回响应头但未返回数据）",
                    timeout.as_secs()
                ))
            })?;

        let Some(first) = first else {
            return Err(ProxyError::ForwardFailed(
                "流式响应在首包到达前结束".to_string(),
            ));
        };

        let first =
            first.map_err(|e| ProxyError::ForwardFailed(format!("读取流式响应首包失败: {e}")))?;

        let replay = futures::stream::once(async move { Ok(first) }).chain(stream);
        Ok(ProxyResponse::streamed(status, headers, replay))
    }

    async fn resolve_claude_api_format(
        &self,
        provider: &Provider,
        body: &Value,
        is_copilot: bool,
    ) -> String {
        if !is_copilot {
            return super::providers::get_claude_api_format(provider).to_string();
        }

        let model = body.get("model").and_then(|value| value.as_str());
        if let Some(model_id) = model {
            if self
                .is_copilot_openai_vendor_model(provider, model_id)
                .await
            {
                return "openai_responses".to_string();
            }
        }

        "openai_chat".to_string()
    }

    /// 用 Copilot live `/models` 列表确认 model ID 真实可用，找不到时按 family 降级。
    /// 命中缓存后是同步的；首次请求或 5 min 缓存过期后会触发一次 HTTP。
    async fn apply_copilot_live_model_resolution(
        &self,
        provider: &Provider,
        body: &mut serde_json::Value,
    ) {
        let Some(model_id) = body.get("model").and_then(|v| v.as_str()) else {
            return;
        };
        let model_id = model_id.to_string();

        let Some(app_handle) = &self.app_handle else {
            return;
        };
        let copilot_state = app_handle.state::<CopilotAuthState>();
        let copilot_auth = copilot_state.0.read().await;
        let account_id = provider
            .meta
            .as_ref()
            .and_then(|m| m.managed_account_id_for("github_copilot"));

        let models_result = match account_id.as_deref() {
            Some(id) => copilot_auth.fetch_models_for_account(id).await,
            None => copilot_auth.fetch_models().await,
        };

        let models = match models_result {
            Ok(m) => m,
            Err(err) => {
                log::debug!("[Copilot] live model list unavailable, skip resolution: {err}");
                return;
            }
        };

        if let Some(resolved) =
            super::providers::copilot_model_map::resolve_against_models(&model_id, &models)
        {
            log::info!("[Copilot] live-model resolve: {model_id} → {resolved}");
            body["model"] = serde_json::Value::String(resolved);
        }
    }

    async fn is_copilot_openai_vendor_model(&self, provider: &Provider, model_id: &str) -> bool {
        let Some(app_handle) = &self.app_handle else {
            log::debug!("[Copilot] AppHandle unavailable, fallback to chat/completions");
            return false;
        };

        let copilot_state = app_handle.state::<CopilotAuthState>();
        let copilot_auth = copilot_state.0.read().await;
        let account_id = provider
            .meta
            .as_ref()
            .and_then(|m| m.managed_account_id_for("github_copilot"));

        let vendor_result = match account_id.as_deref() {
            Some(id) => {
                copilot_auth
                    .get_model_vendor_for_account(id, model_id)
                    .await
            }
            None => copilot_auth.get_model_vendor(model_id).await,
        };

        match vendor_result {
            Ok(Some(vendor)) => vendor.eq_ignore_ascii_case("openai"),
            Ok(None) => {
                log::debug!(
                    "[Copilot] Model vendor unavailable for {model_id}, fallback to chat/completions"
                );
                false
            }
            Err(err) => {
                log::warn!(
                    "[Copilot] Failed to resolve model vendor for {model_id}, fallback to chat/completions: {err}"
                );
                false
            }
        }
    }

    fn categorize_proxy_error(&self, error: &ProxyError) -> ErrorCategory {
        match error {
            // 网络和上游错误：都应该尝试下一个供应商
            ProxyError::Timeout(_) => ErrorCategory::Retryable,
            ProxyError::ForwardFailed(_) => ErrorCategory::Retryable,
            ProxyError::ProviderUnhealthy(_) => ErrorCategory::Retryable,
            // 上游 HTTP 错误：按状态码分桶。
            //
            // 客户端请求自身有问题的状态码无论换哪个 provider 都会被拒绝，
            // 继续轮询只会放大错误率、污染熔断器健康度、浪费配额：
            //   400 Bad Request / 422 Unprocessable Entity   ← 请求体格式或语义错误
            //   405 Method Not Allowed / 406 Not Acceptable  ← 方法或 Accept 错误
            //   413 Payload Too Large / 414 URI Too Long     ← 客户端构造超限
            //   415 Unsupported Media Type                    ← Content-Type 错误
            //   501 Not Implemented                           ← 上游协议确实不支持
            //
            // 其他 4xx（401/403/404/408/409/429/451 等）和全部 5xx 都保留
            // Retryable —— 换一家 provider 可能持有不同的 key、配额、地域或模型映射。
            ProxyError::UpstreamError { status, .. } => match *status {
                400 | 405 | 406 | 413 | 414 | 415 | 422 | 501 => ErrorCategory::NonRetryable,
                // 404 = 模型或端点不存在。换一家 provider 可能映射正确，所以仍触发故障转移，
                // 但不计入熔断器健康度——该供应商对其他模型可能完全正常。
                404 => ErrorCategory::FailoverNeutral,
                _ => ErrorCategory::Retryable,
            },
            // Provider 级配置/转换问题：换一个 Provider 可能就能成功
            ProxyError::ConfigError(_) => ErrorCategory::Retryable,
            ProxyError::TransformError(_) => ErrorCategory::Retryable,
            ProxyError::AuthError(_) => ErrorCategory::Retryable,
            ProxyError::StreamIdleTimeout(_) => ErrorCategory::Retryable,
            // 无可用供应商：所有供应商都试过了，无法重试
            ProxyError::NoAvailableProvider => ErrorCategory::NonRetryable,
            // 其他错误（数据库/内部错误等）：不是换供应商能解决的问题
            _ => ErrorCategory::NonRetryable,
        }
    }
}

/// 从 ProxyError 中提取错误消息
fn extract_error_message(error: &ProxyError) -> Option<String> {
    match error {
        ProxyError::UpstreamError { body, .. } => body.clone(),
        _ => Some(error.to_string()),
    }
}

/// 检测 Provider 是否为 Bedrock（通过 CLAUDE_CODE_USE_BEDROCK 环境变量判断）
fn is_bedrock_provider(provider: &Provider) -> bool {
    provider
        .settings_config
        .get("env")
        .and_then(|e| e.get("CLAUDE_CODE_USE_BEDROCK"))
        .and_then(|v| v.as_str())
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn build_retryable_failure_log(
    provider_name: &str,
    attempted_providers: usize,
    total_providers: usize,
    error: &ProxyError,
) -> (&'static str, String) {
    let error_summary = summarize_proxy_error(error);

    if total_providers <= 1 {
        (
            log_fwd::SINGLE_PROVIDER_FAILED,
            format!("Provider {provider_name} 请求失败: {error_summary}"),
        )
    } else {
        (
            log_fwd::PROVIDER_FAILED_RETRY,
            format!(
                "Provider {provider_name} 失败，继续尝试下一个 ({attempted_providers}/{total_providers}): {error_summary}"
            ),
        )
    }
}

fn build_terminal_failure_log(
    attempted_providers: usize,
    total_providers: usize,
    last_error: Option<&ProxyError>,
) -> Option<(&'static str, String)> {
    if total_providers <= 1 {
        return None;
    }

    let error_summary = last_error
        .map(summarize_proxy_error)
        .unwrap_or_else(|| "未知错误".to_string());

    Some((
        log_fwd::ALL_PROVIDERS_FAILED,
        format!(
            "已尝试 {attempted_providers}/{total_providers} 个 Provider，均失败。最后错误: {error_summary}"
        ),
    ))
}

fn summarize_proxy_error(error: &ProxyError) -> String {
    match error {
        ProxyError::UpstreamError { status, body } => {
            let body_summary = body
                .as_deref()
                .map(summarize_upstream_body)
                .filter(|summary| !summary.is_empty());

            match body_summary {
                Some(summary) => format!("上游 HTTP {status}: {summary}"),
                None => format!("上游 HTTP {status}"),
            }
        }
        ProxyError::Timeout(message) => {
            format!("请求超时: {}", summarize_text_for_log(message, 180))
        }
        ProxyError::ForwardFailed(message) => {
            format!("请求转发失败: {}", summarize_text_for_log(message, 180))
        }
        ProxyError::TransformError(message) => {
            format!("响应转换失败: {}", summarize_text_for_log(message, 180))
        }
        ProxyError::ConfigError(message) => {
            format!("配置错误: {}", summarize_text_for_log(message, 180))
        }
        ProxyError::AuthError(message) => {
            format!("认证失败: {}", summarize_text_for_log(message, 180))
        }
        _ => summarize_text_for_log(&error.to_string(), 180),
    }
}

fn summarize_upstream_body(body: &str) -> String {
    if let Ok(json_body) = serde_json::from_str::<Value>(body) {
        if let Some(message) = extract_json_error_message(&json_body) {
            return summarize_text_for_log(&message, 180);
        }

        if let Ok(compact_json) = serde_json::to_string(&json_body) {
            return summarize_text_for_log(&compact_json, 180);
        }
    }

    summarize_text_for_log(body, 180)
}

fn extract_json_error_message(body: &Value) -> Option<String> {
    let candidates = [
        body.pointer("/error/message"),
        body.pointer("/message"),
        body.pointer("/detail"),
        body.pointer("/error"),
    ];

    candidates
        .into_iter()
        .flatten()
        .find_map(|value| value.as_str().map(ToString::to_string))
}

#[derive(Debug, Clone)]
struct ModelCatalogRoute {
    requested_model: Option<String>,
    upstream_model: Option<String>,
    endpoint: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    auth_header: Option<String>,
    api_format: Option<String>,
}

fn resolve_model_catalog_route(provider: &Provider, body: &Value) -> Option<ModelCatalogRoute> {
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToString::to_string)?;
    let requested_key = canonical_devin_model_key(&requested_model);
    let models = provider
        .settings_config
        .get("modelCatalog")
        .and_then(|catalog| catalog.get("models"))
        .and_then(Value::as_array)?;
    let model_entry = find_devin_model_entry(models, &requested_model, &requested_key)?;
    let route = select_devin_route(model_entry);
    let endpoint = route
        .as_ref()
        .and_then(|route| string_field(route, &["endpoint", "path"]))
        .or_else(|| string_field(model_entry, &["endpoint", "path"]))
        .or_else(|| {
            route
                .as_ref()
                .and_then(|route| {
                    string_field(route, &["apiFormat", "api_format", "wireApi", "wire_api"])
                })
                .or_else(|| {
                    string_field(
                        model_entry,
                        &["apiFormat", "api_format", "wireApi", "wire_api"],
                    )
                })
                .and_then(|format| endpoint_from_devin_format(&format))
        })
        .map(|endpoint| normalize_devin_endpoint(&endpoint));
    let api_format = route
        .as_ref()
        .and_then(|route| string_field(route, &["apiFormat", "api_format", "wireApi", "wire_api"]))
        .or_else(|| {
            string_field(
                model_entry,
                &["apiFormat", "api_format", "wireApi", "wire_api"],
            )
        })
        .or_else(|| {
            endpoint
                .as_deref()
                .and_then(model_catalog_api_format_from_endpoint)
        });
    let base_url = route
        .as_ref()
        .and_then(|route| string_field(route, &["baseUrl", "base_url", "host", "url"]))
        .or_else(|| string_field(model_entry, &["baseUrl", "base_url", "host", "url"]))
        .map(|url| normalize_model_catalog_base_url(&url));
    let api_key = route
        .as_ref()
        .and_then(|route| string_field(route, &["apiKey", "api_key", "key"]))
        .or_else(|| string_field(model_entry, &["apiKey", "api_key", "key"]));
    let upstream_model = string_field(model_entry, &["upstreamModel", "upstream_model"])
        .or_else(|| string_field(model_entry, &["model"]));

    if base_url.is_none() && api_key.is_none() && endpoint.is_none() && upstream_model.is_none() {
        return None;
    }

    Some(ModelCatalogRoute {
        requested_model: Some(requested_model),
        upstream_model,
        endpoint,
        base_url,
        api_key,
        auth_header: route
            .as_ref()
            .and_then(|route| string_field(route, &["authHeader", "auth_header"]))
            .or_else(|| string_field(model_entry, &["authHeader", "auth_header"])),
        api_format,
    })
}

fn model_catalog_api_format_from_endpoint(endpoint: &str) -> Option<String> {
    if is_chat_completions_endpoint(endpoint) {
        Some("openai_chat".to_string())
    } else if is_responses_endpoint(endpoint) {
        Some("openai_responses".to_string())
    } else if is_messages_endpoint(endpoint) {
        Some("anthropic".to_string())
    } else {
        None
    }
}

fn normalize_model_catalog_base_url(url: &str) -> String {
    url.trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string()
}

#[derive(Debug, Clone)]
pub(crate) struct DevinResolvedRoute {
    pub(crate) requested_model: Option<String>,
    pub(crate) upstream_model: Option<String>,
    pub(crate) endpoint: String,
    pub(crate) name: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) api_key: Option<String>,
    pub(crate) auth_header: Option<String>,
    pub(crate) extra_headers: Vec<(String, String)>,
    pub(crate) responses_codex_compat: bool,
    pub(crate) responses_fast_mode: bool,
    pub(crate) thinking_enabled: Option<bool>,
}

pub(crate) fn resolve_devin_model_route(
    provider: &Provider,
    body: &Value,
) -> Option<DevinResolvedRoute> {
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToString::to_string)
        .or_else(|| super::providers::codex_provider_upstream_model(provider));

    let models = provider
        .settings_config
        .get("modelCatalog")
        .and_then(|catalog| catalog.get("models"))
        .and_then(Value::as_array)?;

    let requested_key = requested_model
        .as_deref()
        .map(canonical_devin_model_key)
        .unwrap_or_default();

    let exact_model_entry = requested_model
        .as_ref()
        .and_then(|model| find_devin_model_entry(models, model, &requested_key));
   // 非主模型（小模型）请求不 fallback 到主模型条目，避免小模型被
   // 路由到昂贵的上游（如 glm-5.2），产生意外费用。
   // 小模型请求若无法在当前供应商解析路由则返回 None，请求会带着原始
   // 模型名发往当前供应商（上游通常返回 404/400 触发故障转移）。
    // picpi 是代理聚合服务，上游自行处理模型路由，保留原有 fallback 行为
    let is_non_primary = requested_model
       .as_deref()
       .is_some_and(|m| !is_devin_primary_model_alias(m));
    let is_picpi = is_picpi_devin_catalog(models, provider);
    let model_entry = if is_non_primary && !is_picpi {
       exact_model_entry.or_else(|| find_devin_small_model_entry(models))
   } else {
       exact_model_entry
           .or_else(|| {
               requested_model
                   .as_deref()
                   .filter(|model| !is_devin_primary_model_alias(model))
                   .and_then(|_| find_devin_small_model_entry(models))
           })
           .or_else(|| find_devin_fallback_model_entry(models, provider))
           .or_else(|| models.first())
   };
   let model_entry = model_entry?;

    let endpoint = resolve_devin_model_endpoint(model_entry, provider)?;
    let upstream_model = devin_upstream_model(model_entry, provider);
    let route = select_devin_route(model_entry);
    let mut extra_headers =
        header_fields(model_entry, &["headers", "extraHeaders", "extra_headers"]);
    if let Some(route) = route.as_ref() {
        merge_header_fields(
            &mut extra_headers,
            header_fields(route, &["headers", "extraHeaders", "extra_headers"]),
        );
    }
    let responses_codex_compat =
        resolve_devin_responses_codex_compat(model_entry, route.as_ref(), provider);
    let responses_fast_mode =
        resolve_devin_responses_fast_mode(model_entry, route.as_ref(), provider);
    let thinking_enabled = resolve_devin_thinking_enabled(model_entry, route.as_ref(), provider);

    let base_url_raw = route
        .as_ref()
        .and_then(|route| string_field(route, &["baseUrl", "base_url", "host", "url"]))
        .or_else(|| string_field(model_entry, &["baseUrl", "base_url", "host", "url"]));

    let base_url = base_url_raw.map(|url| {
        let trimmed = url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .to_string();
        log::debug!("[Devin] baseUrl: raw='{}' → trimmed='{}'", url, trimmed);
        trimmed
    });

    Some(DevinResolvedRoute {
        requested_model,
        upstream_model,
        endpoint,
        name: route
            .as_ref()
            .and_then(|route| string_field(route, &["name"]))
            .or_else(|| string_field(model_entry, &["routeName", "route_name"])),
        base_url,
        api_key: route
            .as_ref()
            .and_then(|route| string_field(route, &["apiKey", "api_key", "key"]))
            .or_else(|| string_field(model_entry, &["apiKey", "api_key", "key"])),
        auth_header: route
            .as_ref()
            .and_then(|route| string_field(route, &["authHeader", "auth_header"]))
            .or_else(|| string_field(model_entry, &["authHeader", "auth_header"])),
        extra_headers,
        responses_codex_compat,
        responses_fast_mode,
        thinking_enabled,
    })
}

fn find_devin_model_entry<'a>(
    models: &'a [Value],
    requested_model: &str,
    requested_key: &str,
) -> Option<&'a Value> {
    let requested_exact_key = normalize_devin_model_key(requested_model);
    let exact = models.iter().find(|entry| {
        entry
            .get("model")
            .and_then(Value::as_str)
            .map(normalize_devin_model_key)
            .is_some_and(|model| model == requested_exact_key)
    });
    if exact.is_some() {
        return exact;
    }

    models.iter().find(|entry| {
        [
            "model",
            "upstreamModel",
            "upstream_model",
            "displayName",
            "display_name",
            "name",
        ]
        .iter()
        .any(|key| {
            entry
                .get(*key)
                .and_then(Value::as_str)
                .map(canonical_devin_model_key)
                .is_some_and(|key| key == requested_key)
        })
    })
}

fn find_devin_small_model_entry(models: &[Value]) -> Option<&Value> {
    models.iter().find(|entry| {
        entry
            .get("routes")
            .and_then(Value::as_array)
            .is_some_and(|routes| {
                routes.iter().any(|route| {
                    string_field(route, &["name"]).as_deref() == Some("devin-small-model")
                })
            })
    })
}

fn is_devin_primary_model_alias(model: &str) -> bool {
    matches!(
        normalize_devin_model_key(model).as_str(),
        "swe_1_6_slow"
            | "claude_sonnet_4_thinking_byok"
            | "claude_sonnet_4_byok"
            | "claude_opus_4_thinking_byok"
            | "claude_opus_4_byok"
            | "model_claude_4_sonnet_thinking_byok"
            | "model_claude_4_sonnet_byok"
            | "model_claude_4_opus_thinking_byok"
            | "model_claude_4_opus_byok"
    )
}

fn find_devin_fallback_model_entry<'a>(
    models: &'a [Value],
    provider: &Provider,
) -> Option<&'a Value> {
    if is_picpi_devin_catalog(models, provider) {
        if let Some(entry) = models.iter().find(|entry| {
            resolve_devin_model_endpoint(entry, provider)
                .as_deref()
                .is_some_and(is_responses_endpoint)
                && devin_upstream_model(entry, provider).is_some()
        }) {
            return Some(entry);
        }
    }

    models
        .iter()
        .find(|entry| devin_upstream_model(entry, provider).is_some())
}

fn is_picpi_devin_catalog(models: &[Value], provider: &Provider) -> bool {
    string_field(
        &provider.settings_config,
        &["baseUrl", "base_url", "host", "url"],
    )
    .is_some_and(|url| is_picpi_devin_upstream(&url))
        || models.iter().any(|entry| {
            string_field(entry, &["baseUrl", "base_url", "host", "url"])
                .is_some_and(|url| is_picpi_devin_upstream(&url))
                || select_devin_route(entry).is_some_and(|route| {
                    string_field(&route, &["baseUrl", "base_url", "host", "url"])
                        .is_some_and(|url| is_picpi_devin_upstream(&url))
                })
        })
}

fn devin_upstream_model(model_entry: &Value, provider: &Provider) -> Option<String> {
    let model = string_field(model_entry, &["model"]);
    string_field(model_entry, &["upstreamModel", "upstream_model"])
        .or_else(|| {
            model
                .as_deref()
                .filter(|model| is_devin_windsurf_model_alias(model))
                .and_then(|_| super::providers::codex_provider_upstream_model(provider))
        })
        .or(model)
        .or_else(|| super::providers::codex_provider_upstream_model(provider))
}

fn resolve_devin_model_endpoint(model_entry: &Value, provider: &Provider) -> Option<String> {
    string_field(model_entry, &["endpoint", "path"])
        .or_else(|| {
            string_field(
                model_entry,
                &["apiFormat", "api_format", "wireApi", "wire_api"],
            )
            .and_then(|format| endpoint_from_devin_format(&format))
        })
        .or_else(|| {
            provider
                .meta
                .as_ref()
                .and_then(|meta| meta.api_format.as_deref())
                .and_then(endpoint_from_devin_format)
        })
        .or_else(|| {
            string_field(
                &provider.settings_config,
                &["apiFormat", "api_format", "wireApi", "wire_api"],
            )
            .and_then(|format| endpoint_from_devin_format(&format))
        })
        .map(|endpoint| normalize_devin_endpoint(&endpoint))
}

fn endpoint_from_devin_format(format: &str) -> Option<String> {
    match normalize_devin_model_key(format).as_str() {
        "anthropic" | "anthropic_messages" | "messages" | "claude" => {
            Some("/v1/messages".to_string())
        }
        "openai_chat" | "chat" | "chat_completions" => Some("/v1/chat/completions".to_string()),
        "openai_responses" | "responses" => Some("/v1/responses".to_string()),
        _ => None,
    }
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

fn select_devin_route(model_entry: &Value) -> Option<Value> {
    let mut routes: Vec<Value> = model_entry
        .get("routes")
        .and_then(Value::as_array)
        .map(|routes| {
            routes
                .iter()
                .filter(|route| {
                    route
                        .get("enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(true)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    routes.sort_by(|a, b| {
        let priority_a = a.get("priority").and_then(Value::as_i64).unwrap_or(100);
        let priority_b = b.get("priority").and_then(Value::as_i64).unwrap_or(100);
        priority_a
            .cmp(&priority_b)
            .then_with(|| string_field(a, &["name"]).cmp(&string_field(b, &["name"])))
    });

    routes.into_iter().next()
}

fn resolve_devin_responses_codex_compat(
    model_entry: &Value,
    route: Option<&Value>,
    provider: &Provider,
) -> bool {
    bool_field(
        route.unwrap_or(&Value::Null),
        &[
            "responsesCodexCompat",
            "responses_codex_compat",
            "codexCompat",
            "codex_compat",
        ],
    )
    .or_else(|| {
        bool_field(
            model_entry,
            &[
                "responsesCodexCompat",
                "responses_codex_compat",
                "codexCompat",
                "codex_compat",
            ],
        )
    })
    .or_else(|| {
        bool_field(
            &provider.settings_config,
            &[
                "responsesCodexCompat",
                "responses_codex_compat",
                "codexCompat",
                "codex_compat",
            ],
        )
    })
    .unwrap_or_else(|| {
        let explicit_mode = string_field(
            route.unwrap_or(&Value::Null),
            &["responsesMode", "responses_mode"],
        )
        .or_else(|| string_field(model_entry, &["responsesMode", "responses_mode"]))
        .or_else(|| {
            string_field(
                &provider.settings_config,
                &["responsesMode", "responses_mode"],
            )
        });

        if explicit_mode.is_some_and(|mode| {
            matches!(
                normalize_devin_model_key(&mode).as_str(),
                "codex" | "codex_compat" | "codex_compatible"
            )
        }) {
            return true;
        }

        let base_url = route
            .and_then(|route| string_field(route, &["baseUrl", "base_url", "host", "url"]))
            .or_else(|| string_field(model_entry, &["baseUrl", "base_url", "host", "url"]))
            .or_else(|| {
                string_field(
                    &provider.settings_config,
                    &["baseUrl", "base_url", "host", "url"],
                )
            });

        base_url.is_some_and(|url| is_picpi_devin_upstream(&url))
    })
}

fn is_picpi_devin_upstream(url: &str) -> bool {
    url.to_ascii_lowercase().contains("picpi.top")
}

fn should_use_direct_http_client_for_upstream(
    provider: &Provider,
    devin_route: Option<&DevinResolvedRoute>,
) -> bool {
    let route_base_url = devin_route.and_then(|route| route.base_url.as_deref());
    let route_name = devin_route.and_then(|route| route.name.as_deref());

    bool_field(
        &provider.settings_config,
        &[
            "useDirectClient",
            "use_direct_client",
            "bypassSystemProxy",
            "bypass_system_proxy",
            "noProxy",
            "no_proxy",
        ],
    )
    .unwrap_or(false)
        || route_base_url.is_some_and(|base| {
            let lower = base.to_ascii_lowercase();
            lower.contains("joycode-api") || lower.contains("picpi.top")
        })
        || route_name.is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("joycode") || lower.contains("pipi") || lower.contains("picpi")
        })
        || {
            let lower = provider.name.to_ascii_lowercase();
            lower.contains("pipi") || lower.contains("picpi")
        }
}

fn resolve_devin_responses_fast_mode(
    model_entry: &Value,
    route: Option<&Value>,
    provider: &Provider,
) -> bool {
    bool_field(
        route.unwrap_or(&Value::Null),
        &[
            "responsesFastMode",
            "responses_fast_mode",
            "codexFastMode",
            "codex_fast_mode",
        ],
    )
    .or_else(|| {
        bool_field(
            model_entry,
            &[
                "responsesFastMode",
                "responses_fast_mode",
                "codexFastMode",
                "codex_fast_mode",
            ],
        )
    })
    .or_else(|| {
        bool_field(
            &provider.settings_config,
            &[
                "responsesFastMode",
                "responses_fast_mode",
                "codexFastMode",
                "codex_fast_mode",
            ],
        )
    })
    .unwrap_or(false)
}

fn resolve_devin_thinking_enabled(
    model_entry: &Value,
    route: Option<&Value>,
    provider: &Provider,
) -> Option<bool> {
    bool_field(
        route.unwrap_or(&Value::Null),
        &[
            "thinkingEnabled",
            "thinking_enabled",
            "enableThinking",
            "enable_thinking",
        ],
    )
    .or_else(|| {
        bool_field(
            model_entry,
            &[
                "thinkingEnabled",
                "thinking_enabled",
                "enableThinking",
                "enable_thinking",
            ],
        )
    })
    .or_else(|| {
        bool_field(
            &provider.settings_config,
            &[
                "thinkingEnabled",
                "thinking_enabled",
                "enableThinking",
                "enable_thinking",
            ],
        )
    })
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

fn header_fields(value: &Value, keys: &[&str]) -> Vec<(String, String)> {
    let Some(headers) = keys
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_object))
    else {
        return Vec::new();
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
            let value = value.trim().to_string();
            if value.is_empty() {
                return None;
            }
            Some((name.to_string(), value))
        })
        .collect()
}

fn merge_header_fields(base: &mut Vec<(String, String)>, patch: Vec<(String, String)>) {
    for (name, value) in patch {
        if let Some(existing) = base
            .iter_mut()
            .find(|(existing, _)| existing.eq_ignore_ascii_case(&name))
        {
            existing.1 = value;
        } else {
            base.push((name, value));
        }
    }
}

fn normalize_devin_model_key(value: &str) -> String {
    let mut normalized = String::new();
    let mut last_was_sep = false;

    for ch in value.trim().chars().flat_map(char::to_lowercase) {
        if ch == '-' || ch == '_' || ch == '.' || ch.is_whitespace() {
            if !last_was_sep && !normalized.is_empty() {
                normalized.push('_');
                last_was_sep = true;
            }
        } else {
            normalized.push(ch);
            last_was_sep = false;
        }
    }

    normalized.trim_matches('_').to_string()
}

fn canonical_devin_model_key(value: &str) -> String {
    let key = normalize_devin_model_key(value);
    match key.as_str() {
        "model_claude_4_sonnet_byok" | "claude_sonnet_4_byok" => "claude_sonnet_4_byok".to_string(),
        "model_claude_4_sonnet_thinking_byok"
        | "claude_sonnet_4_thinking_byok"
        | "claude_sonnet_4_thinking" => "claude_sonnet_4_thinking_byok".to_string(),
        "model_claude_4_opus_byok" | "claude_opus_4_byok" => "claude_opus_4_byok".to_string(),
        "model_claude_4_opus_thinking_byok" | "claude_opus_4_thinking_byok" => {
            "claude_opus_4_thinking_byok".to_string()
        }
        "model_claude_haiku_4_5_byok" | "claude_haiku_4_5_byok" => {
            "claude_haiku_4_5_byok".to_string()
        }
        "model_claude_haiku_4_5" | "claude_haiku_4_5" => "claude_haiku_4_5".to_string(),
        "model_gpt_4o" | "gpt_4o" => "gpt_4o".to_string(),
        "model_gpt_4o_mini" | "gpt_4o_mini" => "gpt_4o_mini".to_string(),
        "model_gpt_5_nano" | "gpt_5_nano" => "gpt_5_nano".to_string(),
        "model_google_gemini_2_5_flash" | "google_gemini_2_5_flash" | "gemini_2_5_flash" => {
            "model_google_gemini_2_5_flash".to_string()
        }
        _ => key,
    }
}

fn is_devin_windsurf_model_alias(value: &str) -> bool {
    matches!(
        canonical_devin_model_key(value).as_str(),
        "claude_sonnet_4_byok"
            | "claude_sonnet_4_thinking_byok"
            | "claude_opus_4_byok"
            | "claude_opus_4_thinking_byok"
            | "claude_haiku_4_5"
            | "claude_haiku_4_5_byok"
            | "model_swe_1_5"
            | "model_swe_1_5_slow"
            | "model_chat_11121"
            | "claude_sonnet_4_6_thinking"
            | "claude_opus_4_6_thinking"
            | "model_google_gemini_2_5_flash"
            | "model_google_gemini_2_5_pro"
            | "gpt_5_nano"
            | "model_private_11"
            | "gpt_4o"
            | "gpt_4o_mini"
            | "gpt_5_4_low"
            | "gpt_5_4_high"
            | "gpt_5_4_xhigh"
            | "gpt_5_4_xhigh_priority"
            | "swe_1_6_slow"
            | "swe_1_6_fast"
    )
}

fn devin_route_auth_headers(
    route: &DevinResolvedRoute,
) -> Result<Vec<(http::HeaderName, http::HeaderValue)>, ProxyError> {
    let api_key = route
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .ok_or_else(|| ProxyError::AuthError("Devin route API key is empty".to_string()))?;

    let auth_header = route
        .auth_header
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| {
            if is_messages_endpoint(&route.endpoint) {
                "x-api-key".to_string()
            } else {
                "bearer".to_string()
            }
        });

    if matches!(
        auth_header.as_str(),
        "x-api-key" | "x_api_key" | "anthropic"
    ) {
        let value = http::HeaderValue::from_str(api_key)
            .map_err(|e| ProxyError::AuthError(format!("Invalid Devin x-api-key: {e}")))?;
        return Ok(vec![(http::HeaderName::from_static("x-api-key"), value)]);
    }

    let bearer = format!("Bearer {api_key}");
    let value = http::HeaderValue::from_str(&bearer)
        .map_err(|e| ProxyError::AuthError(format!("Invalid Devin bearer token: {e}")))?;
    Ok(vec![(http::header::AUTHORIZATION, value)])
}

fn model_catalog_route_auth_headers(
    route: &ModelCatalogRoute,
) -> Result<Vec<(http::HeaderName, http::HeaderValue)>, ProxyError> {
    let api_key = route
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .ok_or_else(|| ProxyError::AuthError("Model catalog route API key is empty".to_string()))?;

    let auth_header = route
        .auth_header
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| {
            if route.endpoint.as_deref().is_some_and(is_messages_endpoint) {
                "x-api-key".to_string()
            } else {
                "bearer".to_string()
            }
        });

    if matches!(
        auth_header.as_str(),
        "x-api-key" | "x_api_key" | "anthropic"
    ) {
        let value = http::HeaderValue::from_str(api_key)
            .map_err(|e| ProxyError::AuthError(format!("Invalid route x-api-key: {e}")))?;
        return Ok(vec![(http::HeaderName::from_static("x-api-key"), value)]);
    }

    let bearer = format!("Bearer {api_key}");
    let value = http::HeaderValue::from_str(&bearer)
        .map_err(|e| ProxyError::AuthError(format!("Invalid route bearer token: {e}")))?;
    Ok(vec![(http::header::AUTHORIZATION, value)])
}

fn append_anthropic_beta_tokens(parts: &mut Vec<String>, value: &str) {
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if !parts.iter().any(|existing| existing == token) {
            parts.push(token.to_string());
        }
    }
}

fn is_connect_header_for_json_upstream(header_name: &str) -> bool {
    if header_name.eq_ignore_ascii_case("content-type")
        || header_name.eq_ignore_ascii_case("content-encoding")
        || header_name.eq_ignore_ascii_case("accept")
        || header_name.eq_ignore_ascii_case("te")
    {
        return true;
    }

    let lower = header_name.to_ascii_lowercase();
    lower.starts_with("connect-") || lower.starts_with("grpc-")
}

#[allow(clippy::too_many_arguments)]
fn should_rebuild_connect_headers_for_json_upstream(
    app_type: &AppType,
    has_devin_route: bool,
    devin_messages_to_chat: bool,
    devin_messages_to_responses: bool,
    devin_route_to_responses: bool,
    devin_upstream_is_responses: bool,
    devin_upstream_is_messages: bool,
    codex_model_catalog_to_messages: bool,
) -> bool {
    codex_model_catalog_to_messages
        || (matches!(app_type, AppType::Devin)
            && has_devin_route
            && (devin_messages_to_chat
                || devin_messages_to_responses
                || devin_route_to_responses
                || devin_upstream_is_responses
                || devin_upstream_is_messages))
}

fn is_joycode_provider(provider: &Provider, base_url: &str) -> bool {
    let provider_type = provider
        .meta
        .as_ref()
        .and_then(|meta| meta.provider_type.as_deref())
        .unwrap_or_default();
    if !base_url.to_ascii_lowercase().contains("joycode-api.jd.com") {
        return false;
    }
    let haystack = format!(
        "{} {} {} {} {}",
        provider.id, provider.name, base_url, provider_type, provider.settings_config
    )
    .to_ascii_lowercase();
    haystack.contains("joycode") || haystack.contains("joycode-api.jd.com")
}

fn build_joycode_color_gateway_url(function_id: &str) -> String {
    let timestamp_ms = chrono::Utc::now().timestamp_millis();
    let params = vec![
        ("appid", "joycode_ide".to_string()),
        ("functionId", function_id.to_string()),
        ("t", timestamp_ms.to_string()),
    ];
    let sign = joycode_color_gateway_sign(&params);
    let mut url =
        url::Url::parse("https://api-ai.jd.com/api").expect("JoyCode gateway URL is valid");
    {
        let mut query = url.query_pairs_mut();
        for (key, value) in &params {
            query.append_pair(key, value);
        }
        query.append_pair("sign", &sign);
    }
    url.to_string()
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

fn collect_joycode_provider_headers(provider: &Provider) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    collect_joycode_headers_from_json(&provider.settings_config, &mut headers);

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
                collect_joycode_headers_from_toml(&content, &mut headers);
                break;
            }
        }
    }

    promote_joycode_cookie_header_pairs(&mut headers);
    headers
}

fn collect_joycode_headers_from_json(value: &Value, headers: &mut Vec<(String, String)>) {
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
    let cookie = headers
        .iter()
        .rev()
        .find(|(key, _)| {
            key.eq_ignore_ascii_case("cookie")
                || key.eq_ignore_ascii_case("joycode_cookie")
                || key.eq_ignore_ascii_case("joycodeCookie")
        })
        .map(|(_, value)| value.clone());

    if let Some(cookie) = cookie {
        headers.push(("cookie".to_string(), cookie.clone()));
        if let Some(pt_key) = extract_cookie_value(&cookie, "pt_key") {
            headers.push(("pt_key".to_string(), pt_key.clone()));
            headers.push(("x-pt-key".to_string(), pt_key));
        }
    }
}

fn extract_joycode_auth_params(headers: &[(String, String)]) -> Option<(String, String, String)> {
    let pt_key = headers
        .iter()
        .find(|(key, _)| {
            key.eq_ignore_ascii_case("ptKey")
                || key.eq_ignore_ascii_case("ptkey")
                || key.eq_ignore_ascii_case("pt_key")
                || key.eq_ignore_ascii_case("x-pt-key")
        })
        .map(|(_, value)| value.clone())
        .or_else(|| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case("cookie"))
                .and_then(|(_, value)| extract_cookie_value(value, "pt_key"))
        })?;

    let login_type = headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("loginType"))
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| "ERP".to_string());
    let tenant = headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("tenant"))
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| "JD".to_string());

    Some((pt_key, login_type, tenant))
}

fn append_joycode_auth_headers(
    headers: &mut http::HeaderMap,
    auth_params: Option<&(String, String, String)>,
) -> usize {
    let Some((pt_key, login_type, tenant)) = auth_params else {
        return 0;
    };

    let mut injected_count = 0;
    if let Ok(value) = http::HeaderValue::from_str(pt_key) {
        headers.insert(http::HeaderName::from_static("ptkey"), value);
        injected_count += 1;
        log::debug!("[JoyCode] Injected ptkey");
    }
    if let Ok(value) = http::HeaderValue::from_str(login_type) {
        headers.insert(http::HeaderName::from_static("logintype"), value);
        injected_count += 1;
        log::debug!("[JoyCode] Injected logintype={}", login_type);
    }
    if let Ok(value) = http::HeaderValue::from_str(tenant) {
        headers.insert(http::HeaderName::from_static("tenant"), value);
        injected_count += 1;
        log::debug!("[JoyCode] Injected tenant={}", tenant);
    }

    injected_count
}

fn extract_cookie_value(cookie: &str, name: &str) -> Option<String> {
    cookie.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        (key == name).then(|| value.to_string())
    })
}

fn devin_request_to_anthropic_messages(body: Value) -> Result<Value, ProxyError> {
    if value_is_anthropic_messages_request(&body) {
        return Ok(body);
    }

    let chat_body = if value_is_openai_responses_request(&body) {
        super::providers::transform_codex_chat::responses_to_chat_completions(body)?
    } else {
        body
    };

    super::providers::transform::openai_chat_request_to_anthropic(chat_body)
}

fn codex_request_to_anthropic_messages(body: Value) -> Result<Value, ProxyError> {
    if value_is_anthropic_messages_request(&body) {
        return Ok(body);
    }

    let chat_body = if value_is_openai_responses_request(&body) {
        super::providers::transform_codex_chat::responses_to_chat_completions(body)?
    } else {
        body
    };

    super::providers::transform::openai_chat_request_to_anthropic(chat_body)
        .map(normalize_anthropic_temperature_for_thinking)
}

const DEVIN_CACHE_MAX_TEXT_CHARS: usize = 12_000;
const DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS: usize = 2_000;
const DEVIN_RECENT_MESSAGES_WITHOUT_COMPACTION: usize = 1;
const DEVIN_STABLE_HISTORY_CACHE_POINTS: usize = 2;

fn prepare_devin_anthropic_body(
    body: Value,
    provider: &Provider,
    route: Option<&DevinResolvedRoute>,
) -> (Value, Option<String>) {
    let mut body = compact_devin_context(body);
    inject_devin_anthropic_cache_control(&mut body);
    let cache_key = build_devin_prompt_cache_key(&body, provider, route);
    log_devin_cache_plan(&body, cache_key.as_deref());
    (body, cache_key)
}

fn compact_devin_context(mut body: Value) -> Value {
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        let before_chars = canonical_json_string(&Value::Array(messages.clone()))
            .chars()
            .count();
        let compact_until = messages
            .len()
            .saturating_sub(DEVIN_RECENT_MESSAGES_WITHOUT_COMPACTION);
        for (index, message) in messages.iter_mut().enumerate() {
            if index >= compact_until {
                continue;
            }
            compact_devin_message(message);
        }
        let after_chars = canonical_json_string(&Value::Array(messages.clone()))
            .chars()
            .count();
        log_devin_compaction_stats("anthropic", messages, before_chars, after_chars);
    }
    body
}

fn compact_devin_openai_chat_context(mut body: Value) -> Value {
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        let before_chars = canonical_json_string(&Value::Array(messages.clone()))
            .chars()
            .count();
        let compact_until = messages
            .len()
            .saturating_sub(DEVIN_RECENT_MESSAGES_WITHOUT_COMPACTION);
        for (index, message) in messages.iter_mut().enumerate() {
            if index >= compact_until {
                continue;
            }
            compact_devin_openai_chat_message(message);
        }
        let after_chars = canonical_json_string(&Value::Array(messages.clone()))
            .chars()
            .count();
        log_devin_compaction_stats("openai_chat", messages, before_chars, after_chars);
    }
    body
}

fn compact_devin_openai_chat_message(message: &mut Value) {
    let role = message.get("role").and_then(Value::as_str).unwrap_or("");
    if role == "tool" {
        compact_devin_openai_chat_content(message.get_mut("content"));
        return;
    }

    let Some(content) = message.get_mut("content") else {
        return;
    };
    let Some(parts) = content.as_array_mut() else {
        return;
    };

    for part in parts {
        if part.get("type").and_then(Value::as_str) == Some("tool_result") {
            compact_devin_tool_result(part);
        }
    }
}

fn compact_devin_openai_chat_content(content: Option<&mut Value>) {
    let Some(content) = content else {
        return;
    };

    match content {
        Value::String(text) => {
            *text = compact_long_devin_text(
                text,
                DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                "tool_result",
                Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                true,
            );
        }
        Value::Array(parts) => {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    part["text"] = Value::String(compact_long_devin_text(
                        text,
                        DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                        "tool_result",
                        Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                        true,
                    ));
                }
            }
        }
        other => {
            let serialized = canonical_json_string(other);
            *other = Value::String(compact_long_devin_text(
                &serialized,
                DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                "tool_result",
                Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                true,
            ));
        }
    }
}

fn compact_devin_message(message: &mut Value) {
    let Some(content) = message.get_mut("content") else {
        return;
    };

    if content.as_str().is_some() {
        return;
    }

    let Some(blocks) = content.as_array_mut() else {
        return;
    };

    let mut seen_tool_results = std::collections::BTreeSet::new();
    blocks.retain_mut(|block| {
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
        match block_type {
            "text" => true,
            "tool_result" => {
                let signature = canonical_json_string(&serde_json::json!({
                    "tool_use_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                    "is_error": block.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                }));
                if !seen_tool_results.insert(signature) {
                    return false;
                }
                compact_devin_tool_result(block);
                true
            }
            "tool_use" => {
                compact_devin_tool_use(block);
                true
            }
            "image" => {
                compact_devin_image(block);
                true
            }
            _ => true,
        }
    });
}

fn compact_devin_tool_result(block: &mut Value) {
    let Some(content) = block.get("content").cloned() else {
        return;
    };

    match content {
        Value::String(text) => {
            block["content"] = Value::String(compact_long_devin_text(
                &text,
                DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                "tool_result",
                Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                true,
            ));
        }
        Value::Array(mut parts) => {
            for part in &mut parts {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        part["text"] = Value::String(compact_long_devin_text(
                            text,
                            DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                            "tool_result",
                            Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                            true,
                        ));
                    }
                } else if part.get("type").and_then(Value::as_str) == Some("image") {
                    compact_devin_image(part);
                }
            }
            block["content"] = Value::Array(parts);
        }
        other => {
            let serialized = canonical_json_string(&other);
            block["content"] = Value::String(compact_long_devin_text(
                &serialized,
                DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS * 2,
                "tool_result",
                Some(DEVIN_CACHE_TOOL_RESULT_EDGE_CHARS),
                true,
            ));
        }
    }
}

fn compact_devin_tool_use(block: &mut Value) {
    let Some(input) = block.get("input").cloned() else {
        return;
    };
    let serialized = canonical_json_string(&input);
    if serialized.chars().count() <= DEVIN_CACHE_MAX_TEXT_CHARS {
        return;
    }
    block["input"] = serde_json::json!({
        "_compressed": true,
        "chars": serialized.chars().count(),
        "sha256": short_sha256_hex(serialized.as_bytes()),
        "preview": compact_long_devin_text(
            &serialized,
            DEVIN_CACHE_MAX_TEXT_CHARS,
            "tool_use_input",
            None,
            false,
        ),
    });
}

fn compact_devin_image(block: &mut Value) {
    let media_type = block
        .get("media_type")
        .or_else(|| block.get("mediaType"))
        .or_else(|| block.pointer("/source/media_type"))
        .or_else(|| block.pointer("/source/mediaType"))
        .and_then(Value::as_str)
        .map(ToString::to_string);

    if let Some(object) = block.as_object_mut() {
        object.clear();
        object.insert("type".to_string(), Value::String("image".to_string()));
        if let Some(media_type) = media_type {
            object.insert("media_type".to_string(), Value::String(media_type));
        }
    }
}

fn compact_long_devin_text(
    value: &str,
    max_chars: usize,
    label: &str,
    edge_chars: Option<usize>,
    preserve_whitespace: bool,
) -> String {
    let normalized = normalize_devin_cache_text(value, preserve_whitespace);
    let char_count = normalized.chars().count();
    if char_count <= max_chars {
        return normalized;
    }

    let head_chars = edge_chars.unwrap_or_else(|| std::cmp::max(200, max_chars * 6 / 10));
    let tail_chars = edge_chars.unwrap_or_else(|| std::cmp::max(200, max_chars * 3 / 10));
    let head = normalized.chars().take(head_chars).collect::<String>();
    let tail = normalized
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!(
        "[{label} elided: chars={char_count}, sha256={}]\n{head}\n[...elided...]\n{tail}",
        short_sha256_hex(normalized.as_bytes())
    )
}

fn normalize_devin_cache_text(value: &str, preserve_whitespace: bool) -> String {
    let mut text = value.to_string();
    for (regex, replacement) in [
        (iso_timestamp_regex(), "[timestamp]"),
        (request_id_regex(), "$1[id]"),
        (uuid_regex(), "[uuid]"),
        (call_id_regex(), "call_[id]"),
        (toolu_id_regex(), "toolu_[id]"),
        (msg_id_regex(), "msg_[id]"),
        (resp_id_regex(), "resp_[id]"),
        (tmp_path_regex(), "/tmp/[path]"),
        (var_folders_regex(), "/var/folders/[path]"),
    ] {
        text = regex.replace_all(&text, replacement).into_owned();
    }
    if preserve_whitespace {
        text.trim().to_string()
    } else {
        whitespace_regex()
            .replace_all(text.trim(), " ")
            .into_owned()
    }
}

fn inject_devin_anthropic_cache_control(body: &mut Value) {
    inject_anthropic_system_cache_control(body);
    inject_anthropic_tools_cache_control(body);
    inject_stable_history_cache_control(body);
}

fn inject_anthropic_system_cache_control(body: &mut Value) -> bool {
    let Some(system) = body.get_mut("system") else {
        return false;
    };

    if let Some(text) = system.as_str().map(ToString::to_string) {
        if !text.trim().is_empty() {
            *system = serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": { "type": "ephemeral" }
            }]);
            return true;
        }
        return false;
    }

    if let Some(blocks) = system.as_array_mut() {
        if let Some(block) = blocks.iter_mut().find(|block| {
            block
                .get("type")
                .and_then(Value::as_str)
                .is_none_or(|kind| kind == "text")
                && block
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
        }) {
            block["cache_control"] = serde_json::json!({ "type": "ephemeral" });
            return true;
        }
    }

    false
}

fn inject_anthropic_tools_cache_control(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };

    let Some(tool) = tools.last_mut() else {
        return false;
    };

    tool["cache_control"] = serde_json::json!({ "type": "ephemeral" });
    true
}

fn inject_stable_history_cache_control(body: &mut Value) -> bool {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };

    if messages.len() < 2 {
        return false;
    }

    let mut injected = 0usize;
    for message in messages.iter_mut().rev().skip(1) {
        if message_has_devin_cache_control(message) {
            injected += 1;
        } else if inject_message_cache_control(message) {
            injected += 1;
        }
        if injected >= DEVIN_STABLE_HISTORY_CACHE_POINTS {
            return true;
        }
    }

    injected > 0
}

fn inject_message_cache_control(message: &mut Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };

    if let Some(text) = content.as_str().map(ToString::to_string) {
        if text.trim().is_empty() {
            return false;
        }
        *content = serde_json::json!([{
            "type": "text",
            "text": text,
            "cache_control": { "type": "ephemeral" }
        }]);
        return true;
    }

    let Some(blocks) = content.as_array_mut() else {
        return false;
    };

    for block in blocks.iter_mut().rev() {
        if is_devin_cacheable_message_block(block) {
            block["cache_control"] = serde_json::json!({ "type": "ephemeral" });
            return true;
        }
    }

    false
}

fn is_devin_cacheable_message_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("text" | "tool_result" | "tool_use")
    )
}

fn log_devin_cache_plan(body: &Value, cache_key: Option<&str>) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let message_summaries = messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            format!(
                "{}:{}:{}:{}",
                index,
                message.get("role").and_then(Value::as_str).unwrap_or("?"),
                if message_has_devin_cache_control(message) {
                    "cache"
                } else {
                    "fresh"
                },
                if message_has_devin_compacted_content(message) {
                    "compact"
                } else {
                    "raw"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    log::debug!(
        "[DevinCache] cache_key={}, system={}, tools={}, messages={}, plan=[{}]",
        cache_key
            .map(|key| format!(
                "present(len={},hash={})",
                key.len(),
                short_sha256_hex(key.as_bytes())
            ))
            .unwrap_or_else(|| "absent".to_string()),
        short_value_hash(body.get("system")),
        short_value_hash(body.get("tools")),
        messages.len(),
        message_summaries
    );
}

fn log_devin_compaction_stats(
    format: &str,
    messages: &[Value],
    before_chars: usize,
    after_chars: usize,
) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    let compacted_messages = messages
        .iter()
        .filter(|message| message_has_devin_compacted_content(message))
        .count();
    let cache_messages = messages
        .iter()
        .filter(|message| message_has_devin_cache_control(message))
        .count();
    let saved_chars = before_chars.saturating_sub(after_chars);
    let saved_percent = if before_chars > 0 {
        saved_chars as f64 * 100.0 / before_chars as f64
    } else {
        0.0
    };

    log::debug!(
        "[DevinCompact] format={}, messages={}, compacted={}, cache_marks={}, chars={} -> {} (-{}, {:.1}%)",
        format,
        messages.len(),
        compacted_messages,
        cache_messages,
        before_chars,
        after_chars,
        saved_chars,
        saved_percent
    );
}

fn message_has_devin_cache_control(message: &Value) -> bool {
    message
        .get("content")
        .is_some_and(value_has_devin_cache_control)
}

fn value_has_devin_cache_control(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.contains_key("cache_control")
                || object.values().any(value_has_devin_cache_control)
        }
        Value::Array(values) => values.iter().any(value_has_devin_cache_control),
        _ => false,
    }
}

fn message_has_devin_compacted_content(message: &Value) -> bool {
    message
        .get("content")
        .is_some_and(value_has_devin_compacted_content)
}

fn value_has_devin_compacted_content(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains(" elided:") || text.contains("\"_compressed\":true"),
        Value::Object(object) => {
            object.get("_compressed").and_then(Value::as_bool) == Some(true)
                || object.values().any(value_has_devin_compacted_content)
        }
        Value::Array(values) => values.iter().any(value_has_devin_compacted_content),
        _ => false,
    }
}

fn build_devin_prompt_cache_key(
    body: &Value,
    provider: &Provider,
    route: Option<&DevinResolvedRoute>,
) -> Option<String> {
    let model = route
        .and_then(|route| route.upstream_model.as_deref())
        .or_else(|| body.get("model").and_then(Value::as_str))
        .filter(|model| !model.trim().is_empty())?;
    let system_hash = short_value_hash(body.get("system"));
    let tools_hash = short_value_hash(body.get("tools"));
    let conversation_scope = devin_prompt_cache_conversation_scope(body);
    let raw_key = format!(
        "workspace:cc-switch|agent:devin-v3|provider:{}|route:{}|model:{}|tools:{}|system:{}|scope:{}",
        provider.id,
        route
            .and_then(|route| route.name.as_deref())
            .unwrap_or("default"),
        model,
        tools_hash,
        system_hash,
        conversation_scope
    );
    Some(format!(
        "ccsw-devin-{}",
        short_sha256_hex(raw_key.as_bytes())
    ))
}

fn devin_prompt_cache_conversation_scope(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return "no_messages".to_string();
    };

    let stable_prefix = messages.iter().take(4).cloned().collect::<Vec<_>>();
    if stable_prefix.is_empty() {
        return "empty_messages".to_string();
    }

    short_sha256_hex(canonical_json_string(&Value::Array(stable_prefix)).as_bytes())
}

fn inject_devin_responses_cache_key(
    body: &mut Value,
    provider: &Provider,
    route: Option<&DevinResolvedRoute>,
) {
    if body.get("prompt_cache_key").is_some() {
        return;
    }
    let cache_key = build_devin_prompt_cache_key(body, provider, route);
    if let Some(cache_key) = cache_key {
        body["prompt_cache_key"] = Value::String(cache_key);
        body["prompt_cache_retention"] = Value::String("24h".to_string());
    }
}

fn inject_devin_chat_cache_key(
    body: &mut Value,
    provider: &Provider,
    route: Option<&DevinResolvedRoute>,
) {
    if body.get("prompt_cache_key").is_some() {
        return;
    }
    let cache_key = build_devin_prompt_cache_key(body, provider, route);
    if let Some(cache_key) = cache_key {
        body["prompt_cache_key"] = Value::String(cache_key);
        body["prompt_cache_retention"] = Value::String("24h".to_string());
    }
}

fn iso_timestamp_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b").unwrap())
}

fn request_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\b((?:request|req|trace|session|cursor|message|turn)[_-]?id\s*[:=]\s*)[A-Za-z0-9._:-]+").unwrap()
    })
}

fn uuid_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b").unwrap()
    })
}

fn call_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\bcall_[A-Za-z0-9_-]+\b").unwrap())
}

fn toolu_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\btoolu_[A-Za-z0-9_-]+\b").unwrap())
}

fn msg_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\bmsg_[A-Za-z0-9_-]+\b").unwrap())
}

fn resp_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\bresp_[A-Za-z0-9_-]+\b").unwrap())
}

fn tmp_path_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"/tmp/[^\s)]+").unwrap())
}

fn var_folders_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"/var/folders/[^\s)]+").unwrap())
}

fn whitespace_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\s+").unwrap())
}

fn normalize_anthropic_temperature_for_thinking(mut body: Value) -> Value {
    let thinking_enabled =
        body.get("thinking")
            .and_then(Value::as_object)
            .is_some_and(|thinking| {
                !matches!(
                    thinking.get("type").and_then(Value::as_str),
                    Some("disabled")
                )
            });

    if !thinking_enabled {
        return body;
    }

    let temperature_is_one = body
        .get("temperature")
        .and_then(Value::as_f64)
        .is_some_and(|temperature| (temperature - 1.0).abs() < f64::EPSILON);

    if !temperature_is_one {
        if let Some(object) = body.as_object_mut() {
            object.remove("temperature");
        }
    }

    body
}

fn apply_devin_codex_responses_compat(mut body: Value, codex_fast_mode: bool) -> Value {
    body["store"] = Value::Bool(false);
    if codex_fast_mode {
        body["service_tier"] = Value::String("priority".to_string());
    }

    const REASONING_MARKER: &str = "reasoning.encrypted_content";
    let mut include = body
        .get("include")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !include
        .iter()
        .any(|value| value.as_str() == Some(REASONING_MARKER))
    {
        include.push(Value::String(REASONING_MARKER.to_string()));
    }
    body["include"] = Value::Array(include);

    if let Some(object) = body.as_object_mut() {
        object.remove("max_output_tokens");
        object.remove("temperature");
        object.remove("top_p");
        object
            .entry("instructions".to_string())
            .or_insert_with(|| Value::String(String::new()));
        object
            .entry("tools".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        object
            .entry("parallel_tool_calls".to_string())
            .or_insert_with(|| Value::Bool(false));
        object.insert("stream".to_string(), Value::Bool(true));
    }

    body
}

fn value_is_openai_responses_request(body: &Value) -> bool {
    body.get("input").is_some()
        || body.get("instructions").is_some()
        || body.get("previous_response_id").is_some()
        || body.get("max_output_tokens").is_some()
}

fn value_is_openai_chat_request(body: &Value) -> bool {
    body.get("messages").and_then(Value::as_array).is_some()
        && !value_is_anthropic_messages_request(body)
}

fn value_is_anthropic_messages_request(body: &Value) -> bool {
    if body.get("_cc_switch_canonical_api").and_then(Value::as_str) == Some("anthropic_messages") {
        return true;
    }

    body.get("system").is_some()
        || body.get("anthropic_version").is_some()
        || body.get("stop_sequences").is_some()
        || body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| tools.iter().any(|tool| tool.get("input_schema").is_some()))
        || body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message
                        .get("content")
                        .and_then(Value::as_array)
                        .is_some_and(|content| {
                            content.iter().any(|part| {
                                matches!(
                                    part.get("type").and_then(Value::as_str),
                                    Some(
                                        "tool_use"
                                            | "tool_result"
                                            | "thinking"
                                            | "redacted_thinking"
                                            | "server_tool_use"
                                            | "web_search_tool_result"
                                    )
                                ) || part.get("source").is_some()
                                    || part.get("input").is_some()
                                    || part.get("tool_use_id").is_some()
                            })
                        })
                })
            })
}

fn replace_endpoint_path_preserve_query(original: &str, replacement_path: &str) -> String {
    let (_, query) = split_endpoint_and_query(original);
    match query {
        Some(query) if !query.is_empty() => format!("{replacement_path}?{query}"),
        _ => replacement_path.to_string(),
    }
}

fn endpoint_path_matches(endpoint: &str, matches: &[&str]) -> bool {
    let (path, _) = split_endpoint_and_query(endpoint);
    matches.iter().any(|candidate| path == *candidate)
}

fn is_messages_endpoint(endpoint: &str) -> bool {
    let (path, _) = split_endpoint_and_query(endpoint);
    endpoint_path_matches(endpoint, &["/messages", "/v1/messages"]) || path.ends_with("/messages")
}

fn is_chat_completions_endpoint(endpoint: &str) -> bool {
    let (path, _) = split_endpoint_and_query(endpoint);
    endpoint_path_matches(endpoint, &["/chat/completions", "/v1/chat/completions"])
        || path.ends_with("/chat/completions")
}

fn is_responses_endpoint(endpoint: &str) -> bool {
    let (path, _) = split_endpoint_and_query(endpoint);
    endpoint_path_matches(
        endpoint,
        &[
            "/responses",
            "/v1/responses",
            "/responses/compact",
            "/v1/responses/compact",
        ],
    ) || path.ends_with("/responses")
        || path.ends_with("/responses/compact")
}

fn split_endpoint_and_query(endpoint: &str) -> (&str, Option<&str>) {
    endpoint
        .split_once('?')
        .map_or((endpoint, None), |(path, query)| (path, Some(query)))
}

fn strip_beta_query(query: Option<&str>) -> Option<String> {
    let filtered = query.map(|query| {
        query
            .split('&')
            .filter(|pair| !pair.is_empty() && !pair.starts_with("beta="))
            .collect::<Vec<_>>()
            .join("&")
    });

    match filtered.as_deref() {
        Some("") | None => None,
        Some(_) => filtered,
    }
}

fn is_claude_messages_path(path: &str) -> bool {
    matches!(path, "/v1/messages" | "/claude/v1/messages")
}

fn rewrite_codex_responses_endpoint_to_chat(endpoint: &str) -> (String, Option<String>) {
    let (_path, query) = split_endpoint_and_query(endpoint);
    let passthrough_query = query.map(ToString::to_string);
    let target_path = "/chat/completions";
    let rewritten = match passthrough_query.as_deref() {
        Some(query) if !query.is_empty() => format!("{target_path}?{query}"),
        _ => target_path.to_string(),
    };

    (rewritten, passthrough_query)
}

fn rewrite_codex_chat_endpoint_to_responses(endpoint: &str) -> (String, Option<String>) {
    let (_path, query) = split_endpoint_and_query(endpoint);
    let passthrough_query = query.map(ToString::to_string);
    let target_path = "/responses";
    let rewritten = match passthrough_query.as_deref() {
        Some(query) if !query.is_empty() => format!("{target_path}?{query}"),
        _ => target_path.to_string(),
    };

    (rewritten, passthrough_query)
}

fn rewrite_claude_transform_endpoint(
    endpoint: &str,
    api_format: &str,
    is_copilot: bool,
    body: &Value,
) -> (String, Option<String>) {
    let (path, query) = split_endpoint_and_query(endpoint);
    let passthrough_query = if is_claude_messages_path(path) {
        strip_beta_query(query)
    } else {
        query.map(ToString::to_string)
    };

    if !is_claude_messages_path(path) {
        return (endpoint.to_string(), passthrough_query);
    }

    if api_format == "gemini_native" {
        let model =
            super::providers::transform_gemini::extract_gemini_model(body).unwrap_or("unknown");
        // Accept both bare ids (`gemini-2.5-pro`) and the resource-name
        // form (`models/gemini-2.5-pro`) that Gemini SDKs emit. See
        // `normalize_gemini_model_id` for rationale.
        let model = super::gemini_url::normalize_gemini_model_id(model);
        let is_stream = body
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let target_path = if is_stream {
            format!("/v1beta/models/{model}:streamGenerateContent")
        } else {
            format!("/v1beta/models/{model}:generateContent")
        };

        let rewritten_query = merge_query_params(
            passthrough_query.as_deref(),
            if is_stream { Some("alt=sse") } else { None },
        );

        let rewritten = match rewritten_query.as_deref() {
            Some(query) if !query.is_empty() => format!("{target_path}?{query}"),
            _ => target_path,
        };

        return (rewritten, rewritten_query);
    }

    let target_path = if is_copilot && api_format == "openai_responses" {
        "/v1/responses"
    } else if is_copilot {
        "/chat/completions"
    } else if api_format == "openai_responses" {
        "/v1/responses"
    } else {
        "/v1/chat/completions"
    };

    let rewritten = match passthrough_query.as_deref() {
        Some(query) if !query.is_empty() => format!("{target_path}?{query}"),
        _ => target_path.to_string(),
    };

    (rewritten, passthrough_query)
}

fn merge_query_params(base_query: Option<&str>, extra_param: Option<&str>) -> Option<String> {
    let mut params: Vec<String> = base_query
        .into_iter()
        .flat_map(|query| query.split('&'))
        .filter(|pair| !pair.is_empty())
        .filter(|pair| !pair.starts_with("alt="))
        .map(ToString::to_string)
        .collect();

    if let Some(extra_param) = extra_param {
        params.push(extra_param.to_string());
    }

    if params.is_empty() {
        None
    } else {
        Some(params.join("&"))
    }
}

fn append_query_to_full_url(base_url: &str, query: Option<&str>) -> String {
    match query {
        Some(query) if !query.is_empty() => {
            if base_url.contains('?') {
                format!("{base_url}&{query}")
            } else {
                format!("{base_url}?{query}")
            }
        }
        _ => base_url.to_string(),
    }
}

fn build_codex_oauth_session_headers(
    session_id: &str,
) -> Vec<(http::HeaderName, http::HeaderValue)> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Vec::new();
    }

    let mut headers = Vec::new();
    if let Ok(value) = http::HeaderValue::from_str(session_id) {
        headers.push((http::HeaderName::from_static("session_id"), value.clone()));
        headers.push((http::HeaderName::from_static("x-client-request-id"), value));
    }

    let window_id = format!("{session_id}:0");
    if let Ok(value) = http::HeaderValue::from_str(&window_id) {
        headers.push((http::HeaderName::from_static("x-codex-window-id"), value));
    }

    headers
}

fn reject_proxy_placeholder_for_managed_account_upstream(
    url: &str,
    headers: &http::HeaderMap,
) -> Result<(), ProxyError> {
    if !is_managed_account_upstream_url(url) || !headers_contain_proxy_placeholder(headers) {
        return Ok(());
    }

    Err(ProxyError::AuthError(
        "Managed account proxy auth was not resolved; PROXY_MANAGED must not be sent upstream"
            .to_string(),
    ))
}

fn is_managed_account_upstream_url(url: &str) -> bool {
    let Ok(uri) = url.parse::<http::Uri>() else {
        return false;
    };

    let Some(host) = uri.host().map(str::to_ascii_lowercase) else {
        return false;
    };

    host == "githubcopilot.com"
        || host.ends_with(".githubcopilot.com")
        || (host == "chatgpt.com" && uri.path().starts_with("/backend-api/codex"))
}

fn headers_contain_proxy_placeholder(headers: &http::HeaderMap) -> bool {
    headers.values().any(|value| {
        value
            .to_str()
            .map(|value| value.contains(PROXY_AUTH_PLACEHOLDER))
            .unwrap_or(false)
    })
}

fn should_preserve_exact_header_case(
    adapter_name: &str,
    provider: &Provider,
    resolved_claude_api_format: Option<&str>,
    is_copilot: bool,
) -> bool {
    if matches!(adapter_name, "Codex" | "Gemini") {
        return false;
    }

    if is_copilot || provider.is_codex_oauth() {
        return false;
    }

    matches!(resolved_claude_api_format, None | Some("anthropic"))
}

fn is_streaming_request(endpoint: &str, body: &Value, headers: &axum::http::HeaderMap) -> bool {
    if body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return true;
    }

    if endpoint.contains("streamGenerateContent") || endpoint.contains("alt=sse") {
        return true;
    }

    headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .map(|accept| accept.contains("text/event-stream"))
        .unwrap_or(false)
}

#[cfg(test)]
fn should_force_identity_encoding(
    endpoint: &str,
    body: &Value,
    headers: &axum::http::HeaderMap,
) -> bool {
    is_streaming_request(endpoint, body, headers)
}

fn map_reqwest_send_error(error: reqwest::Error) -> ProxyError {
    let details = describe_reqwest_error(&error);
    if error.is_timeout() {
        ProxyError::Timeout(format!("请求超时: {details}"))
    } else if error.is_connect() {
        ProxyError::ForwardFailed(format!("连接失败: {details}"))
    } else {
        ProxyError::ForwardFailed(details)
    }
}

fn describe_reqwest_error(error: &reqwest::Error) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(err) = source {
        let text = err.to_string();
        if !text.is_empty() && !parts.iter().any(|part| part == &text) {
            parts.push(text);
        }
        source = err.source();
    }
    format!(
        "{} (timeout={}, connect={}, request={})",
        parts.join("; caused by: "),
        error.is_timeout(),
        error.is_connect(),
        error.is_request()
    )
}

fn log_reqwest_send_error(
    error: &reqwest::Error,
    app_type: &str,
    provider_name: &str,
    url: &str,
    body_len: usize,
) {
    log::warn!(
        "[Forwarder] reqwest send failed app_type={} provider={} url={} body_bytes={} error={}",
        app_type,
        provider_name,
        url,
        body_len,
        describe_reqwest_error(error)
    );
}

fn summarize_text_for_log(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = normalized.trim();

    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    let truncated: String = trimmed.chars().take(max_chars).collect();
    let truncated = truncated.trim_end();
    format!("{truncated}...")
}

fn apply_local_proxy_body_overrides(
    body: &mut Value,
    overrides: &LocalProxyRequestOverrides,
) -> bool {
    let Some(override_body) = overrides.body.as_ref() else {
        return false;
    };

    if !override_body.is_object() {
        log::warn!("[LocalProxyOverrides] Ignoring body override because it is not an object");
        return false;
    }

    merge_json_override(body, override_body)
}

fn merge_json_override(target: &mut Value, patch: &Value) -> bool {
    merge_json_override_inner(target, patch, true)
}

fn merge_json_override_inner(target: &mut Value, patch: &Value, is_top_level: bool) -> bool {
    match (target, patch) {
        (Value::Object(target_map), Value::Object(patch_map)) => {
            let mut changed = false;
            for (key, patch_value) in patch_map {
                if is_top_level && key == "stream" {
                    log::warn!(
                        "[LocalProxyOverrides] Ignoring body override for protected field: stream"
                    );
                    continue;
                }
                match target_map.get_mut(key) {
                    Some(target_value) => {
                        changed |= merge_json_override_inner(target_value, patch_value, false);
                    }
                    None => {
                        target_map.insert(key.clone(), patch_value.clone());
                        changed = true;
                    }
                }
            }
            changed
        }
        (target_value, patch_value) => {
            if target_value == patch_value {
                false
            } else {
                *target_value = patch_value.clone();
                true
            }
        }
    }
}

fn apply_local_proxy_header_overrides(
    headers: &mut http::HeaderMap,
    overrides: Option<&LocalProxyRequestOverrides>,
    is_copilot: bool,
) {
    if is_copilot {
        return;
    }

    let Some(header_overrides) = overrides.map(|overrides| &overrides.headers) else {
        return;
    };

    for (raw_name, raw_value) in header_overrides {
        let header_name = raw_name.trim().to_ascii_lowercase();
        if header_name.is_empty() {
            log::warn!("[LocalProxyOverrides] Ignoring header override with empty name");
            continue;
        }

        let Ok(name) = http::HeaderName::from_bytes(header_name.as_bytes()) else {
            log::warn!("[LocalProxyOverrides] Ignoring invalid header override name: {raw_name}");
            continue;
        };

        if is_protected_local_proxy_override_header(&name) {
            log::debug!(
                "[LocalProxyOverrides] Ignoring protected header override: {}",
                name.as_str()
            );
            continue;
        }

        let Ok(value) = http::HeaderValue::from_str(raw_value) else {
            log::warn!(
                "[LocalProxyOverrides] Ignoring invalid header override value for {}",
                name.as_str()
            );
            continue;
        };

        headers.insert(name, value);
    }
}

fn is_protected_local_proxy_override_header(name: &http::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "host"
            | "content-length"
            | "transfer-encoding"
            | "connection"
            | "proxy-authorization"
            | "proxy-authenticate"
            | "te"
            | "trailer"
            | "upgrade"
            | "accept-encoding"
            | "content-type"
            | "authorization"
            | "x-api-key"
            | "x-goog-api-key"
            | "chatgpt-account-id"
            | "session_id"
            | "x-client-request-id"
            | "x-codex-window-id"
            | "x-forwarded-host"
            | "x-forwarded-port"
            | "x-forwarded-proto"
            | "forwarded"
            | "cf-connecting-ip"
            | "cf-ipcountry"
            | "cf-ray"
            | "cf-visitor"
            | "true-client-ip"
            | "fastly-client-ip"
            | "x-azure-clientip"
            | "x-azure-fdid"
            | "x-azure-ref"
            | "akamai-origin-hop"
            | "x-akamai-config-log-detail"
            | "x-request-id"
            | "x-correlation-id"
            | "x-trace-id"
            | "x-amzn-trace-id"
            | "x-b3-traceid"
            | "x-b3-spanid"
            | "x-b3-parentspanid"
            | "x-b3-sampled"
            | "traceparent"
            | "tracestate"
    )
}

fn prepare_upstream_request_body(request_body: Value) -> Value {
    canonicalize_value(filter_private_params_with_whitelist(request_body, &[]))
}

fn strip_devin_route_thinking(mut body: Value) -> Value {
    if let Some(object) = body.as_object_mut() {
        object.remove("thinking");
    }
    super::copilot_optimizer::strip_thinking_blocks(body)
}

fn log_prompt_cache_trace(
    app_type: &AppType,
    provider: &Provider,
    endpoint: &str,
    api_format: Option<&str>,
    body: &Value,
    session_client_provided: bool,
) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    let prompt_cache_key = body
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .map(|key| {
            format!(
                "present(len={},hash={})",
                key.len(),
                short_sha256_hex(key.as_bytes())
            )
        })
        .unwrap_or_else(|| "absent".to_string());
    let store = body
        .get("store")
        .map(value_for_log)
        .unwrap_or_else(|| "absent".to_string());
    let stream = body
        .get("stream")
        .map(value_for_log)
        .unwrap_or_else(|| "absent".to_string());

    log::debug!(
        "[CacheTrace] app={}, provider={}, endpoint={}, api_format={}, session_client_provided={}, prompt_cache_key={}, store={}, stream={}, instructions_hash={}, tools_hash={}, input_hash={}, include_hash={}, body_hash={}",
        app_type.as_str(),
        provider.id,
        endpoint,
        api_format.unwrap_or("native"),
        session_client_provided,
        prompt_cache_key,
        store,
        stream,
        short_value_hash(body.get("instructions")),
        short_value_hash(body.get("tools")),
        short_value_hash(body.get("input")),
        short_value_hash(body.get("include")),
        short_value_hash(Some(body)),
    );
}

fn log_devin_chat_prefix_trace(provider: &Provider, endpoint: &str, body: &Value) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return;
    };

    let mut checkpoints = vec![1usize, 2, 4, 8];
    checkpoints.extend((16..=messages.len()).step_by(16));
    for count in [
        messages.len().saturating_sub(4),
        messages.len().saturating_sub(2),
        messages.len().saturating_sub(1),
        messages.len(),
    ] {
        if count > 0 {
            checkpoints.push(count);
        }
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();

    let prefix_hashes = checkpoints
        .iter()
        .copied()
        .filter(|count| *count <= messages.len())
        .map(|count| {
            let prefix = messages.iter().take(count).cloned().collect::<Vec<_>>();
            format!(
                "{}:{}",
                count,
                short_sha256_hex(canonical_json_string(&Value::Array(prefix)).as_bytes())
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let chunk_hashes = messages
        .chunks(16)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let start = chunk_index * 16;
            let end = start + chunk.len();
            format!(
                "{}-{}:{}",
                start,
                end.saturating_sub(1),
                short_sha256_hex(canonical_json_string(&Value::Array(chunk.to_vec())).as_bytes())
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let role_runs = messages
        .iter()
        .map(
            |message| match message.get("role").and_then(Value::as_str).unwrap_or("?") {
                "system" => "s",
                "user" => "u",
                "assistant" => "a",
                "tool" => "t",
                _ => "?",
            },
        )
        .collect::<Vec<_>>()
        .join("");

    let tail = messages
        .iter()
        .enumerate()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(index, message)| {
            format!(
                "{}:{}:{}",
                index,
                message.get("role").and_then(Value::as_str).unwrap_or("?"),
                short_value_hash(Some(message))
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    log::debug!(
        "[DevinCacheTrace] provider={} endpoint={} model={} messages={} roles_hash={} roles_tail={} body_hash={} prefix=[{}] chunks=[{}] tail=[{}]",
        provider.id,
        endpoint,
        body.get("model").and_then(Value::as_str).unwrap_or(""),
        messages.len(),
        short_sha256_hex(role_runs.as_bytes()),
        role_runs
            .chars()
            .rev()
            .take(32)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>(),
        short_value_hash(Some(body)),
        prefix_hashes,
        chunk_hashes,
        tail,
    );
}

fn value_for_log(value: &Value) -> String {
    match value {
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Null => "null".to_string(),
        Value::Array(values) => format!("array(len={})", values.len()),
        Value::Object(values) => format!("object(len={})", values.len()),
    }
}

async fn restore_sensitive_pseudonyms_in_response(
    response: ProxyResponse,
    rewrite_map: super::sensitive_redaction::SensitiveRewriteMap,
) -> Result<ProxyResponse, ProxyError> {
    let status = response.status();
    let headers = response.headers().clone();

    if response.is_sse() {
        let stream = response.bytes_stream().map(move |chunk| {
            chunk.map(|bytes| restore_sensitive_pseudonyms_in_bytes(bytes, &rewrite_map))
        });
        Ok(ProxyResponse::streamed(status, headers, stream))
    } else {
        let bytes = response.bytes().await?;
        Ok(ProxyResponse::buffered(
            status,
            headers,
            restore_sensitive_pseudonyms_in_bytes(bytes, &rewrite_map),
        ))
    }
}

fn restore_sensitive_pseudonyms_in_bytes(
    bytes: Bytes,
    rewrite_map: &super::sensitive_redaction::SensitiveRewriteMap,
) -> Bytes {
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return bytes;
    };
    let restored = rewrite_map.restore_text(text);
    if restored.as_bytes() == bytes.as_ref() {
        bytes
    } else {
        Bytes::from(restored)
    }
}

fn redact_header_value_for_log(name: &str, value: &str) -> String {
    if name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("cookie")
        || name.eq_ignore_ascii_case("x-api-key")
        || name.eq_ignore_ascii_case("api-key")
        || name.eq_ignore_ascii_case("apikey")
        || name.eq_ignore_ascii_case("x-model-token")
        || name.eq_ignore_ascii_case("ptkey")
        || name.eq_ignore_ascii_case("pt_key")
        || name.eq_ignore_ascii_case("x-pt-key")
    {
        format!("<redacted,len={}>", value.len())
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use crate::provider::LocalProxyRequestOverrides;
    use axum::http::header::{HeaderValue, ACCEPT};
    use axum::http::HeaderMap;
    use bytes::Bytes;
    use http::StatusCode;
    use serde_json::json;
    use std::collections::HashMap;
    use std::time::Duration;

    fn test_provider_with_type(provider_type: Option<&str>) -> Provider {
        Provider {
            id: "provider-1".to_string(),
            name: "Provider 1".to_string(),
            settings_config: json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: provider_type.map(|value| crate::provider::ProviderMeta {
                provider_type: Some(value.to_string()),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        }
    }

    fn test_forwarder(
        non_streaming_timeout: Duration,
        streaming_first_byte_timeout: Duration,
    ) -> RequestForwarder {
        let db = Arc::new(Database::memory().expect("memory db"));

        RequestForwarder {
            router: Arc::new(ProviderRouter::new(db.clone())),
            status: Arc::new(RwLock::new(ProxyStatus::default())),
            current_providers: Arc::new(RwLock::new(HashMap::new())),
            gemini_shadow: Arc::new(GeminiShadowStore::new()),
            codex_chat_history: Arc::new(CodexChatHistoryStore::default()),
            failover_manager: Arc::new(FailoverSwitchManager::new(db)),
            app_handle: None,
            current_provider_id_at_start: String::new(),
            session_id: String::new(),
            session_client_provided: false,
            rectifier_config: RectifierConfig::default(),
            optimizer_config: OptimizerConfig::default(),
            copilot_optimizer_config: CopilotOptimizerConfig::default(),
            non_streaming_timeout,
            streaming_first_byte_timeout,
            max_attempts: 1,
        }
    }

    #[test]
    fn devin_canonical_messages_marker_prevents_double_conversion_and_is_filtered() {
        let body = json!({
            "_cc_switch_canonical_api": "anthropic_messages",
            "model": "gpt-5.5",
            "messages": [{
                "role": "user",
                "content": "ping"
            }],
            "max_tokens": 1024,
            "stream": true
        });

        let canonical = devin_request_to_anthropic_messages(body).unwrap();
        assert_eq!(canonical["messages"][0]["content"], "ping");
        assert_eq!(canonical["_cc_switch_canonical_api"], "anthropic_messages");

        let upstream = prepare_upstream_request_body(canonical);
        assert!(upstream.get("_cc_switch_canonical_api").is_none());
        assert_eq!(upstream["messages"][0]["content"], "ping");
    }

    #[test]
    fn devin_context_prepare_compacts_large_blocks_and_adds_cache_hints() {
        let provider = test_provider_with_type(Some("codex"));
        let older_tool_result = format!(
            "request_id=req-001 2026-06-13T01:02:03Z {} /tmp/older/path/file.txt",
            "o".repeat(9_000)
        );
        let long_tool_result = format!(
            "request_id=req-123 2026-06-14T01:02:03Z {} /tmp/deep/path/file.txt",
            "x".repeat(9_000)
        );
        let latest_tool_result = format!(
            "request_id=req-999 2026-06-15T01:02:03Z {} /tmp/latest/path/file.txt",
            "y".repeat(9_000)
        );
        let long_text = "z".repeat(20_000);
        let body = json!({
            "model": "gpt-5.3-codex",
            "system": "workspace rules",
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_old",
                        "content": older_tool_result
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": long_text
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_abc123",
                        "content": long_tool_result
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_latest",
                        "content": latest_tool_result
                    }]
                }
            ],
            "tools": [{
                "name": "Read",
                "description": "Read files",
                "input_schema": {"type": "object"}
            }]
        });

        let (prepared, cache_key) = prepare_devin_anthropic_body(body, &provider, None);

        assert!(cache_key
            .as_deref()
            .is_some_and(|key| key.starts_with("ccsw-devin-") && key.len() <= 64));
        assert_eq!(
            prepared["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            prepared["tools"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            prepared["messages"][2]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        let old_content = prepared["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap();
        assert!(old_content.contains("[tool_result elided:"));
        assert!(old_content.contains("request_id=[id]"));
        assert!(old_content.contains("[timestamp]"));
        assert!(old_content.contains("/tmp/[path]"));
        let text = prepared["messages"][1]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(text, long_text);
        let stable_history_content = prepared["messages"][2]["content"][0]["content"]
            .as_str()
            .unwrap();
        assert!(stable_history_content.contains("[tool_result elided:"));
        assert!(stable_history_content.contains("request_id=[id]"));
        assert!(stable_history_content.contains("[timestamp]"));
        assert!(stable_history_content.contains("/tmp/[path]"));
        let latest_content = prepared["messages"][3]["content"][0]["content"]
            .as_str()
            .unwrap();
        assert_eq!(latest_content, latest_tool_result);
        assert!(prepared["messages"][3]["content"][0]
            .get("cache_control")
            .is_none());
        assert!(latest_content.contains("request_id=req-999"));
        assert!(latest_content.contains("2026-06-15T01:02:03Z"));
        assert!(latest_content.contains("/tmp/latest/path/file.txt"));
    }

    #[test]
    fn devin_prompt_cache_key_uses_stable_conversation_prefix() {
        let provider = test_provider_with_type(Some("codex"));
        let base = json!({
            "model": "gpt-5.5",
            "system": "workspace rules",
            "messages": [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second prompt"},
                {"role": "assistant", "content": "second answer"},
                {"role": "user", "content": "third prompt"},
                {"role": "assistant", "content": "third answer"}
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}]
        });
        let mut with_tail = base.clone();
        with_tail["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role": "user", "content": "new tail"}));
        let mut changed_prefix = base.clone();
        changed_prefix["messages"][0]["content"] = json!("different prompt");

        let base_key = build_devin_prompt_cache_key(&base, &provider, None).unwrap();
        let tail_key = build_devin_prompt_cache_key(&with_tail, &provider, None).unwrap();
        let changed_key = build_devin_prompt_cache_key(&changed_prefix, &provider, None).unwrap();

        assert_eq!(base_key, tail_key);
        assert_ne!(base_key, changed_key);
    }

    #[test]
    fn devin_route_thinking_strip_is_explicit_opt_out_only() {
        let body = json!({
            "thinking": {"type": "enabled", "budget_tokens": 12000},
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "visible"}
                ]}
            ]
        });

        let stripped = strip_devin_route_thinking(body.clone());
        assert!(stripped.get("thinking").is_none());
        assert_eq!(
            stripped["messages"][0]["content"].as_array().unwrap().len(),
            1
        );
        assert_eq!(stripped["messages"][0]["content"][0]["type"], "text");

        assert!(body.get("thinking").is_some());
        assert_eq!(body["messages"][0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn anthropic_thinking_drops_incompatible_temperature() {
        let body = json!({
            "model": "Claude-Sonnet-4.6-hq",
            "thinking": {"type": "enabled", "budget_tokens": 12000},
            "temperature": 0,
            "messages": [{ "role": "user", "content": "ping" }],
            "max_tokens": 1024
        });

        let normalized = normalize_anthropic_temperature_for_thinking(body);

        assert!(normalized.get("thinking").is_some());
        assert!(normalized.get("temperature").is_none());
    }

    #[test]
    fn anthropic_temperature_is_preserved_without_thinking() {
        let body = json!({
            "model": "Claude-Sonnet-4.6-hq",
            "temperature": 0,
            "messages": [{ "role": "user", "content": "ping" }],
            "max_tokens": 1024
        });

        let normalized = normalize_anthropic_temperature_for_thinking(body);

        assert_eq!(normalized["temperature"], 0);
    }

    #[test]
    fn joycode_auth_params_extract_cookie_and_inject_required_headers() {
        let headers = vec![(
            "cookie".to_string(),
            "pt_pin=user; pt_key=BJ.secret; qid_uid=1".to_string(),
        )];
        let auth_params = extract_joycode_auth_params(&headers).unwrap();
        let mut outgoing = http::HeaderMap::new();

        let injected = append_joycode_auth_headers(&mut outgoing, Some(&auth_params));

        assert_eq!(injected, 3);
        assert_eq!(outgoing["ptkey"], "BJ.secret");
        assert_eq!(outgoing["logintype"], "ERP");
        assert_eq!(outgoing["tenant"], "JD");
    }

    #[test]
    fn devin_responses_route_can_request_codex_compatible_shape() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [{
                    "model": "MODEL_PRIVATE_11",
                    "upstreamModel": "gpt-5.5",
                    "endpoint": "/v1/responses",
                    "responsesMode": "codex"
                }]
            }
        });
        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_PRIVATE_11" })).unwrap();
        assert!(route.responses_codex_compat);

        let body = json!({
            "model": "gpt-5.5",
            "input": [{
                "role": "user",
                "content": [{ "type": "input_text", "text": "ping" }]
            }],
            "max_output_tokens": 1024,
            "temperature": 0,
            "stream": false
        });
        let compat = apply_devin_codex_responses_compat(body, false);
        assert_eq!(compat["store"], false);
        assert_eq!(compat["stream"], true);
        assert_eq!(compat["tools"], json!([]));
        assert_eq!(compat["parallel_tool_calls"], false);
        assert!(compat["include"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str() == Some("reasoning.encrypted_content")));
        assert!(compat.get("max_output_tokens").is_none());
        assert!(compat.get("temperature").is_none());
    }

    #[test]
    fn devin_picpi_responses_route_uses_codex_compatible_shape() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [{
                    "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                    "upstreamModel": "gpt-5.5",
                    "endpoint": "/v1/responses",
                    "baseUrl": "https://cn.picpi.top"
                }]
            }
        });

        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_CLAUDE_4_SONNET_BYOK" }))
                .unwrap();

        assert!(route.responses_codex_compat);
    }

    #[test]
    fn devin_picpi_route_bypasses_system_proxy() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.name = "pipi".to_string();
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [{
                    "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                    "upstreamModel": "gpt-5.5",
                    "endpoint": "/v1/responses",
                    "baseUrl": "https://cn.picpi.top"
                }]
            }
        });

        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_CLAUDE_4_SONNET_BYOK" }))
                .unwrap();

        assert!(should_use_direct_http_client_for_upstream(
            &provider,
            Some(&route)
        ));
    }

    #[test]
    fn devin_provider_can_explicitly_bypass_system_proxy() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({ "bypassSystemProxy": true });

        assert!(should_use_direct_http_client_for_upstream(&provider, None));
    }

    #[test]
    fn devin_picpi_unknown_alias_falls_back_to_responses_route() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [
                    {
                        "model": "swe-1-6-slow",
                        "upstreamModel": "gpt-5.5",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://cn.picpi.top"
                    },
                    {
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                        "upstreamModel": "gpt-5.5",
                        "endpoint": "/v1/responses",
                        "baseUrl": "https://cn.picpi.top"
                    }
                ]
            }
        });

        let route = resolve_devin_model_route(
            &provider,
            &json!({ "model": "MODEL_GOOGLE_GEMINI_2_5_FLASH" }),
        )
        .unwrap();

        assert_eq!(route.endpoint, "/v1/responses");
        assert_eq!(route.upstream_model.as_deref(), Some("gpt-5.5"));
        assert!(route.responses_codex_compat);
    }

    #[test]
    fn devin_cheap_alias_routes_to_siliconflow_chat() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [
                    {
                        "model": "swe-1-6-slow",
                        "upstreamModel": "gpt-5.5",
                        "endpoint": "/v1/responses",
                        "baseUrl": "https://openai-plus.example"
                    },
                    {
                        "model": "MODEL_GPT_5_NANO",
                        "upstreamModel": "deepseek-ai/DeepSeek-V3.2",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://api.siliconflow.cn",
                        "apiKey": "sk-test",
                        "authHeader": "bearer",
                        "thinkingEnabled": false
                    },
                    {
                        "model": "MODEL_GOOGLE_GEMINI_2_5_FLASH",
                        "upstreamModel": "deepseek-ai/DeepSeek-V3.2",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://api.siliconflow.cn",
                        "apiKey": "sk-test",
                        "authHeader": "bearer",
                        "thinkingEnabled": false
                    }
                ]
            }
        });

        for model in ["MODEL_GPT_5_NANO", "MODEL_GOOGLE_GEMINI_2_5_FLASH"] {
            let route = resolve_devin_model_route(&provider, &json!({ "model": model })).unwrap();

            assert_eq!(route.endpoint, "/v1/chat/completions");
            assert_eq!(
                route.upstream_model.as_deref(),
                Some("deepseek-ai/DeepSeek-V3.2")
            );
            assert_eq!(
                route.base_url.as_deref(),
                Some("https://api.siliconflow.cn")
            );
            assert_eq!(route.thinking_enabled, Some(false));
        }
   }

    #[test]
    fn devin_small_model_no_fallback_to_primary_on_non_picpi() {
        // 非主模型（小模型）请求在没有小模型条目的非 picpi 供应商上
        // 不应 fallback 到主模型条目，避免产生意外费用
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [
                    {
                        "model": "swe-1-6-slow",
                        "upstreamModel": "glm-5.2",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://open.bigmodel.cn"
                    },
                    {
                        "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                        "upstreamModel": "glm-5.2",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://open.bigmodel.cn"
                    }
                ]
            }
        });

        // 小模型请求在当前供应商没有匹配条目，应返回 None（不 fallback 到 glm-5.2）
        let result = resolve_devin_model_route(
            &provider,
            &json!({ "model": "MODEL_GPT_5_NANO" }),
        );
        assert!(
            result.is_none(),
            "Small model request should not fall back to primary model on non-picpi provider"
        );

        // 主模型请求仍应正常路由
        let route = resolve_devin_model_route(
            &provider,
            &json!({ "model": "swe-1-6-slow" }),
        )
        .unwrap();
        assert_eq!(route.upstream_model.as_deref(), Some("glm-5.2"));
    }

    #[test]
    fn devin_unknown_non_primary_model_routes_to_small_model() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
        "modelCatalog": {
            "models": [
                {
                    "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                    "upstreamModel": "gpt-5.5",
                    "endpoint": "/v1/responses",
                    "baseUrl": "https://cn.picpi.top",
                    "routes": [{ "name": "primary" }]
                },
                {
                    "model": "MODEL_GPT_5_NANO",
                    "upstreamModel": "deepseek-ai/DeepSeek-V3.2",
                    "endpoint": "/v1/chat/completions",
                    "baseUrl": "https://api.siliconflow.cn",
                    "apiKey": "sk-test",
                    "authHeader": "bearer",
                    "routes": [{ "name": "devin-small-model" }]
                }
            ]
        }
        });

        let route = resolve_devin_model_route(
            &provider,
            &json!({ "model": "MODEL_CHAT_GPT_4_1_MINI_2025_04_14" }),
        )
        .unwrap();

        assert_eq!(route.endpoint, "/v1/chat/completions");
        assert_eq!(
            route.upstream_model.as_deref(),
            Some("deepseek-ai/DeepSeek-V3.2")
        );
        assert_eq!(
            route.base_url.as_deref(),
            Some("https://api.siliconflow.cn")
        );
    }

    #[test]
    fn devin_exact_model_id_beats_display_name_match() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [
                    {
                        "model": "Claude Sonnet 4 Thinking BYOK",
                        "displayName": "Claude Sonnet 4 Thinking BYOK",
                        "upstreamModel": "gpt-5.5",
                        "endpoint": "/v1/responses",
                        "baseUrl": "https://cn.picpi.top"
                    },
                    {
                        "model": "MODEL_CLAUDE_4_SONNET_THINKING_BYOK",
                        "displayName": "Small model",
                        "upstreamModel": "deepseek-ai/DeepSeek-V3.2",
                        "endpoint": "/v1/chat/completions",
                        "baseUrl": "https://api.siliconflow.cn"
                    }
                ]
            }
        });

        let route = resolve_devin_model_route(
            &provider,
            &json!({ "model": "MODEL_CLAUDE_4_SONNET_THINKING_BYOK" }),
        )
        .unwrap();

        assert_eq!(
            route.upstream_model.as_deref(),
            Some("deepseek-ai/DeepSeek-V3.2")
        );
        assert_eq!(route.endpoint, "/v1/chat/completions");
        assert_eq!(
            route.base_url.as_deref(),
            Some("https://api.siliconflow.cn")
        );
    }

    #[test]
    fn devin_windsurf_messages_body_converts_for_responses_route() {
        let body = json!({
            "model": "GPT 5.3-codex",
            "messages": [{
                "role": "user",
                "content": [{ "type": "text", "text": "ping" }]
            }],
            "max_tokens": 1024,
            "stream": true
        });

        assert!(!value_is_openai_responses_request(&body));
        let responses = crate::proxy::providers::transform_responses::anthropic_to_responses(
            devin_request_to_anthropic_messages(body).unwrap(),
            None,
            true,
            false,
        )
        .unwrap();

        assert!(responses.get("messages").is_none());
        assert!(responses.get("max_tokens").is_none());
        assert_eq!(responses["model"], "GPT 5.3-codex");
        assert_eq!(responses["input"][0]["role"], "user");
        assert_eq!(responses["stream"], true);
    }

    #[test]
    fn codex_responses_body_converts_for_messages_catalog_route() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "input": [{
                "role": "user",
                "content": [{ "type": "input_text", "text": "ping" }]
            }],
            "max_output_tokens": 1024,
            "stream": true
        });

        let messages = codex_request_to_anthropic_messages(body).unwrap();

        assert!(messages.get("input").is_none());
        assert!(messages.get("max_output_tokens").is_none());
        assert_eq!(messages["model"], "claude-sonnet-4-6");
        assert_eq!(messages["messages"][0]["role"], "user");
        assert_eq!(messages["max_tokens"], 1024);
        assert_eq!(messages["stream"], true);
    }

    #[test]
    fn devin_endpoint_detection_accepts_provider_prefixed_paths() {
        assert!(is_messages_endpoint("/api/saas/anthropic/v1/messages"));
        assert!(is_responses_endpoint("/api/saas/openai/v1/responses"));
        assert!(is_chat_completions_endpoint(
            "/api/saas/openai/v2/chat/completions"
        ));
    }

    #[test]
    fn sensitive_header_values_are_redacted_for_logs() {
        assert_eq!(
            redact_header_value_for_log("pt_key", "BJ.secret"),
            "<redacted,len=9>"
        );
        assert_eq!(
            redact_header_value_for_log("x-model-token", "mt_secret"),
            "<redacted,len=9>"
        );
        assert_eq!(
            redact_header_value_for_log("user-agent", "codex_cli_rs/0.80.0"),
            "codex_cli_rs/0.80.0"
        );
    }

    #[test]
    fn devin_route_matches_windsurf_byok_enum_to_display_name() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "config": r#"model_provider = "custom"
model = "gpt-5.5"

[model_providers.custom]
base_url = "https://muyuan.do/v1"
wire_api = "responses""#,
            "modelCatalog": {
                "models": [{
                    "model": "gpt-5.5",
                    "displayName": "Claude Sonnet 4 BYOK",
                    "upstreamModel": "gpt-5.5",
                    "endpoint": "/v1/chat/completions",
                    "baseUrl": "https://muyuan.do"
                }]
            }
        });

        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_CLAUDE_4_SONNET_BYOK" }))
                .unwrap();

        assert_eq!(route.upstream_model.as_deref(), Some("gpt-5.5"));
        assert_eq!(route.endpoint, "/v1/chat/completions");
    }

    #[test]
    fn devin_route_uses_provider_default_for_unmapped_windsurf_alias() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "config": r#"model_provider = "custom"
model = "glm-5.1"

[model_providers.custom]
base_url = "https://llmhub.qzz.io/v1"
wire_api = "responses""#,
            "modelCatalog": {
                "models": [{
                    "model": "MODEL_CLAUDE_4_OPUS_BYOK",
                    "displayName": "Claude Opus 4 BYOK",
                    "endpoint": "/v1/chat/completions",
                    "baseUrl": "https://llmhub.qzz.io"
                }]
            }
        });

        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_CLAUDE_4_OPUS_BYOK" }))
                .unwrap();

        assert_eq!(route.upstream_model.as_deref(), Some("glm-5.1"));
        assert_eq!(route.endpoint, "/v1/chat/completions");
    }

    #[test]
    fn devin_route_merges_model_and_route_headers() {
        let mut provider = test_provider_with_type(Some("openai"));
        provider.settings_config = json!({
            "modelCatalog": {
                "models": [{
                    "model": "MODEL_CLAUDE_4_SONNET_BYOK",
                    "upstreamModel": "claude-sonnet-4-5-20250929",
                    "endpoint": "/v1/messages",
                    "headers": {
                        "anthropic-beta": "context-1m-2025-08-07",
                        "x-model-header": "model"
                    },
                    "routes": [{
                        "name": "primary",
                        "baseUrl": "https://anyrouter.top",
                        "apiKey": "sk-test",
                        "headers": {
                            "x-model-header": "route"
                        }
                    }]
                }]
            }
        });

        let route =
            resolve_devin_model_route(&provider, &json!({ "model": "MODEL_CLAUDE_4_SONNET_BYOK" }))
                .unwrap();

        assert_eq!(
            route.extra_headers,
            vec![
                (
                    "anthropic-beta".to_string(),
                    "context-1m-2025-08-07".to_string()
                ),
                ("x-model-header".to_string(), "route".to_string())
            ]
        );
    }

    #[test]
    fn json_upstream_drops_connect_rpc_headers() {
        for header in [
            "content-type",
            "Content-Encoding",
            "accept",
            "te",
            "connect-content-encoding",
            "connect-timeout-ms",
            "grpc-timeout",
        ] {
            assert!(
                is_connect_header_for_json_upstream(header),
                "{header} should be rebuilt for JSON upstream"
            );
        }

        assert!(!is_connect_header_for_json_upstream("user-agent"));
        assert!(!is_connect_header_for_json_upstream("anthropic-beta"));
        assert!(!is_connect_header_for_json_upstream("x-custom"));
    }

    #[test]
    fn codex_messages_catalog_route_rebuilds_connect_headers() {
        assert!(should_rebuild_connect_headers_for_json_upstream(
            &AppType::Codex,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        ));
        assert!(!should_rebuild_connect_headers_for_json_upstream(
            &AppType::Codex,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        ));
    }

    #[test]
    fn single_provider_retryable_log_uses_single_provider_code() {
        let error = ProxyError::UpstreamError {
            status: 429,
            body: Some(r#"{"error":{"message":"rate limit exceeded"}}"#.to_string()),
        };

        let (code, message) = build_retryable_failure_log("PackyCode-response", 1, 1, &error);

        assert_eq!(code, log_fwd::SINGLE_PROVIDER_FAILED);
        assert!(message.contains("Provider PackyCode-response 请求失败"));
        assert!(message.contains("上游 HTTP 429"));
        assert!(message.contains("rate limit exceeded"));
        assert!(!message.contains("切换下一个"));
    }

    #[test]
    fn multi_provider_retryable_log_keeps_failover_wording() {
        let error = ProxyError::Timeout("upstream timed out after 30s".to_string());

        let (code, message) = build_retryable_failure_log("primary", 1, 3, &error);

        assert_eq!(code, log_fwd::PROVIDER_FAILED_RETRY);
        assert!(message.contains("继续尝试下一个 (1/3)"));
        assert!(message.contains("请求超时"));
    }

    #[test]
    fn single_provider_has_no_terminal_all_failed_log() {
        assert!(build_terminal_failure_log(1, 1, None).is_none());
    }

    #[test]
    fn multi_provider_terminal_log_contains_last_error_summary() {
        let error = ProxyError::ForwardFailed("connection reset by peer".to_string());

        let (code, message) =
            build_terminal_failure_log(2, 2, Some(&error)).expect("expected terminal log");

        assert_eq!(code, log_fwd::ALL_PROVIDERS_FAILED);
        assert!(message.contains("已尝试 2/2 个 Provider，均失败"));
        assert!(message.contains("connection reset by peer"));
    }

    #[test]
    fn summarize_upstream_body_prefers_json_message() {
        let body = json!({
            "error": {
                "message": "invalid_request_error: unsupported field"
            },
            "request_id": "req_123"
        });

        let summary = summarize_upstream_body(&body.to_string());

        assert_eq!(summary, "invalid_request_error: unsupported field");
    }

    #[test]
    fn summarize_text_for_log_collapses_whitespace_and_truncates() {
        let summary = summarize_text_for_log("line1\n\n line2   line3", 12);

        assert_eq!(summary, "line1 line2...");
    }

    #[test]
    fn canonical_json_sorts_object_keys_for_cache_trace_hashes() {
        let left = json!({
            "tools": [
                {
                    "parameters": {
                        "properties": {
                            "b": {"type": "string"},
                            "a": {"type": "number"}
                        },
                        "type": "object"
                    },
                    "name": "lookup"
                }
            ]
        });
        let right = json!({
            "tools": [
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "string"}
                        }
                    }
                }
            ]
        });

        assert_eq!(
            crate::proxy::json_canonical::canonical_json_string(&left),
            crate::proxy::json_canonical::canonical_json_string(&right)
        );
        assert_eq!(
            short_value_hash(Some(&left)),
            short_value_hash(Some(&right))
        );
    }

    #[test]
    fn prepare_upstream_request_body_filters_private_fields_and_canonicalizes_order() {
        let body = json!({
            "z": 1,
            "_internal": "drop",
            "tools": [
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "_id": {
                                "_private_note": "drop",
                                "type": "string"
                            },
                            "b": {"type": "number"},
                            "a": {"type": "string"}
                        }
                    }
                }
            ],
            "a": 2
        });

        let prepared = prepare_upstream_request_body(body);

        assert!(prepared.get("_internal").is_none());
        assert!(prepared["tools"][0]["parameters"]["properties"]
            .get("_id")
            .is_some());
        assert!(prepared["tools"][0]["parameters"]["properties"]["_id"]
            .get("_private_note")
            .is_none());
        assert_eq!(
            serde_json::to_string(&prepared).unwrap(),
            r#"{"a":2,"tools":[{"name":"lookup","parameters":{"properties":{"_id":{"type":"string"},"a":{"type":"string"},"b":{"type":"number"}},"type":"object"}}],"z":1}"#
        );
    }

    #[test]
    fn local_proxy_body_overrides_deep_merge_final_body_without_stream() {
        let mut body = json!({
            "model": "before",
            "stream": false,
            "metadata": {
                "keep": true,
                "temperature": 1
            },
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let overrides = LocalProxyRequestOverrides {
            headers: HashMap::new(),
            body: Some(json!({
                "model": "after",
                "stream": true,
                "metadata": {
                    "temperature": 0.2,
                    "top_p": 0.9
                },
                "messages": []
            })),
        };

        assert!(apply_local_proxy_body_overrides(&mut body, &overrides));

        assert_eq!(body["model"], "after");
        assert_eq!(body["stream"], false);
        assert_eq!(body["metadata"]["keep"], true);
        assert_eq!(body["metadata"]["temperature"], 0.2);
        assert_eq!(body["metadata"]["top_p"], 0.9);
        assert_eq!(body["messages"], json!([]));
    }

    #[test]
    fn local_proxy_header_overrides_replace_allowed_headers_only() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::USER_AGENT,
            http::HeaderValue::from_static("original"),
        );
        headers.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_static("Bearer good"),
        );
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let overrides = LocalProxyRequestOverrides {
            headers: HashMap::from([
                ("User-Agent".to_string(), "custom".to_string()),
                ("X-Test".to_string(), "ok".to_string()),
                ("Authorization".to_string(), "Bearer bad".to_string()),
                ("Content-Type".to_string(), "text/plain".to_string()),
                ("X-Bad".to_string(), "bad\nvalue".to_string()),
            ]),
            body: None,
        };

        apply_local_proxy_header_overrides(&mut headers, Some(&overrides), false);

        assert_eq!(
            headers
                .get(http::header::USER_AGENT)
                .and_then(|value| value.to_str().ok()),
            Some("custom")
        );
        assert_eq!(
            headers
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer good")
        );
        assert_eq!(
            headers
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            headers.get("x-test").and_then(|value| value.to_str().ok()),
            Some("ok")
        );
        assert!(headers.get("x-bad").is_none());
    }

    #[test]
    fn local_proxy_header_overrides_are_skipped_for_copilot() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::USER_AGENT,
            http::HeaderValue::from_static("copilot"),
        );
        let overrides = LocalProxyRequestOverrides {
            headers: HashMap::from([("User-Agent".to_string(), "custom".to_string())]),
            body: None,
        };

        apply_local_proxy_header_overrides(&mut headers, Some(&overrides), true);

        assert_eq!(
            headers
                .get(http::header::USER_AGENT)
                .and_then(|value| value.to_str().ok()),
            Some("copilot")
        );
    }

    #[tokio::test]
    async fn non_streaming_success_is_buffered_before_marking_provider_successful() {
        let forwarder = test_forwarder(Duration::from_secs(1), Duration::from_secs(1));
        let response = ProxyResponse::streamed(
            StatusCode::OK,
            HeaderMap::new(),
            futures::stream::once(async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"{\"ok\":true}"))
            }),
        );

        let prepared = forwarder
            .prepare_success_response_for_failover(response, false)
            .await
            .expect("response should be buffered");

        assert_eq!(
            prepared.bytes().await.unwrap(),
            Bytes::from_static(b"{\"ok\":true}")
        );
    }

    #[tokio::test]
    async fn non_streaming_body_read_error_is_retryable_before_success_record() {
        let forwarder = test_forwarder(Duration::from_secs(1), Duration::from_secs(1));
        let response = ProxyResponse::streamed(
            StatusCode::OK,
            HeaderMap::new(),
            futures::stream::once(async {
                Err::<Bytes, std::io::Error>(std::io::Error::other("body boom"))
            }),
        );

        let err = match forwarder
            .prepare_success_response_for_failover(response, false)
            .await
        {
            Ok(_) => panic!("body read errors should fail the attempt"),
            Err(err) => err,
        };

        assert!(matches!(err, ProxyError::ForwardFailed(_)));
    }

    #[tokio::test]
    async fn streaming_success_primes_first_chunk_and_replays_it() {
        let forwarder = test_forwarder(Duration::from_secs(1), Duration::from_secs(1));
        let response = ProxyResponse::streamed(
            StatusCode::OK,
            HeaderMap::new(),
            futures::stream::iter(vec![
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"first")),
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"second")),
            ]),
        );

        let prepared = forwarder
            .prepare_success_response_for_failover(response, true)
            .await
            .expect("stream should be primed");

        assert_eq!(
            prepared.bytes().await.unwrap(),
            Bytes::from_static(b"firstsecond")
        );
    }

    #[tokio::test]
    async fn streaming_first_chunk_error_is_retryable_before_success_record() {
        let forwarder = test_forwarder(Duration::from_secs(1), Duration::from_secs(1));
        let response = ProxyResponse::streamed(
            StatusCode::OK,
            HeaderMap::new(),
            futures::stream::once(async {
                Err::<Bytes, std::io::Error>(std::io::Error::other("first chunk boom"))
            }),
        );

        let err = match forwarder
            .prepare_success_response_for_failover(response, true)
            .await
        {
            Ok(_) => panic!("first chunk errors should fail the attempt"),
            Err(err) => err,
        };

        assert!(matches!(err, ProxyError::ForwardFailed(_)));
    }

    #[test]
    fn codex_oauth_session_headers_match_codex_cache_identity() {
        let headers = build_codex_oauth_session_headers("session-123");
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(name, value);
        }

        assert_eq!(
            map.get("session_id"),
            Some(&HeaderValue::from_static("session-123"))
        );
        assert_eq!(
            map.get("x-client-request-id"),
            Some(&HeaderValue::from_static("session-123"))
        );
        assert_eq!(
            map.get("x-codex-window-id"),
            Some(&HeaderValue::from_static("session-123:0"))
        );
    }

    #[test]
    fn managed_account_upstream_rejects_proxy_managed_placeholder_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer PROXY_MANAGED"),
        );

        let err = reject_proxy_placeholder_for_managed_account_upstream(
            "https://api.githubcopilot.com/chat/completions",
            &headers,
        )
        .expect_err("placeholder should be rejected before upstream");

        assert!(matches!(
            err,
            ProxyError::AuthError(message) if message.contains("PROXY_MANAGED")
        ));
    }

    #[test]
    fn codex_oauth_upstream_rejects_proxy_managed_placeholder_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer PROXY_MANAGED"),
        );

        let err = reject_proxy_placeholder_for_managed_account_upstream(
            "https://chatgpt.com/backend-api/codex/responses",
            &headers,
        )
        .expect_err("placeholder should be rejected before upstream");

        assert!(matches!(
            err,
            ProxyError::AuthError(message) if message.contains("PROXY_MANAGED")
        ));
    }

    #[test]
    fn non_managed_upstream_allows_proxy_managed_placeholder_guard() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer PROXY_MANAGED"),
        );

        reject_proxy_placeholder_for_managed_account_upstream(
            "https://api.example.com/v1/messages",
            &headers,
        )
        .expect("guard is scoped to managed-account upstreams");
    }

    #[test]
    fn exact_header_case_preserved_for_native_claude_only() {
        let provider = test_provider_with_type(None);

        assert!(should_preserve_exact_header_case(
            "Claude",
            &provider,
            Some("anthropic"),
            false
        ));
        assert!(!should_preserve_exact_header_case(
            "Claude",
            &provider,
            Some("openai_responses"),
            false
        ));
        assert!(!should_preserve_exact_header_case(
            "Codex", &provider, None, false
        ));
        assert!(!should_preserve_exact_header_case(
            "Gemini", &provider, None, false
        ));
    }

    #[test]
    fn exact_header_case_skipped_for_codex_oauth_and_copilot() {
        let codex_oauth = test_provider_with_type(Some("codex_oauth"));
        let copilot = test_provider_with_type(Some("github_copilot"));

        assert!(!should_preserve_exact_header_case(
            "Claude",
            &codex_oauth,
            Some("openai_responses"),
            false
        ));
        assert!(!should_preserve_exact_header_case(
            "Claude",
            &copilot,
            Some("openai_chat"),
            true
        ));
    }

    #[test]
    fn rewrite_claude_transform_endpoint_strips_beta_for_chat_completions() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/v1/messages?beta=true&foo=bar",
            "openai_chat",
            false,
            &json!({ "model": "gpt-5.4" }),
        );

        assert_eq!(endpoint, "/v1/chat/completions?foo=bar");
        assert_eq!(passthrough_query.as_deref(), Some("foo=bar"));
    }

    #[test]
    fn rewrite_claude_transform_endpoint_strips_beta_for_responses() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/claude/v1/messages?beta=true&x-id=1",
            "openai_responses",
            false,
            &json!({ "model": "gpt-5.4" }),
        );

        assert_eq!(endpoint, "/v1/responses?x-id=1");
        assert_eq!(passthrough_query.as_deref(), Some("x-id=1"));
    }

    #[test]
    fn rewrite_codex_responses_endpoint_to_chat_preserves_query() {
        let (endpoint, passthrough_query) =
            rewrite_codex_responses_endpoint_to_chat("/v1/responses?foo=bar");

        assert_eq!(endpoint, "/chat/completions?foo=bar");
        assert_eq!(passthrough_query.as_deref(), Some("foo=bar"));
    }

    #[test]
    fn rewrite_codex_responses_compact_endpoint_to_chat_preserves_query() {
        let (endpoint, passthrough_query) =
            rewrite_codex_responses_endpoint_to_chat("/v1/responses/compact?foo=bar");

        assert_eq!(endpoint, "/chat/completions?foo=bar");
        assert_eq!(passthrough_query.as_deref(), Some("foo=bar"));
    }

    #[test]
    fn rewrite_codex_chat_endpoint_to_responses_preserves_query() {
        let (endpoint, passthrough_query) =
            rewrite_codex_chat_endpoint_to_responses("/v1/chat/completions?foo=bar");

        assert_eq!(endpoint, "/responses?foo=bar");
        assert_eq!(passthrough_query.as_deref(), Some("foo=bar"));
    }

    #[test]
    fn rewrite_claude_transform_endpoint_uses_copilot_path() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/v1/messages?beta=true&x-id=1",
            "anthropic",
            true,
            &json!({ "model": "claude-sonnet-4-6" }),
        );

        assert_eq!(endpoint, "/chat/completions?x-id=1");
        assert_eq!(passthrough_query.as_deref(), Some("x-id=1"));
    }

    #[test]
    fn rewrite_claude_transform_endpoint_uses_copilot_responses_path() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/v1/messages?beta=true&x-id=1",
            "openai_responses",
            true,
            &json!({ "model": "gpt-5.4" }),
        );

        assert_eq!(endpoint, "/v1/responses?x-id=1");
        assert_eq!(passthrough_query.as_deref(), Some("x-id=1"));
    }

    #[test]
    fn rewrite_claude_transform_endpoint_maps_gemini_generate_content() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/v1/messages?beta=true&x-id=1",
            "gemini_native",
            false,
            &json!({ "model": "gemini-2.5-pro" }),
        );

        assert_eq!(
            endpoint,
            "/v1beta/models/gemini-2.5-pro:generateContent?x-id=1"
        );
        assert_eq!(passthrough_query.as_deref(), Some("x-id=1"));
    }

    /// Regression: body.model arriving as the resource-name form
    /// `models/gemini-2.5-pro` must not produce a doubled
    /// `/v1beta/models/models/...` path.
    #[test]
    fn rewrite_claude_transform_endpoint_strips_gemini_model_resource_prefix() {
        let (endpoint, _) = rewrite_claude_transform_endpoint(
            "/v1/messages",
            "gemini_native",
            false,
            &json!({ "model": "models/gemini-2.5-pro" }),
        );

        assert_eq!(endpoint, "/v1beta/models/gemini-2.5-pro:generateContent");
    }

    #[test]
    fn rewrite_claude_transform_endpoint_maps_gemini_streaming() {
        let (endpoint, passthrough_query) = rewrite_claude_transform_endpoint(
            "/v1/messages?beta=true",
            "gemini_native",
            false,
            &json!({ "model": "gemini-2.5-flash", "stream": true }),
        );

        assert_eq!(
            endpoint,
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(passthrough_query.as_deref(), Some("alt=sse"));
    }

    #[test]
    fn append_query_to_full_url_preserves_existing_query_string() {
        let url = append_query_to_full_url("https://relay.example/api?foo=bar", Some("x-id=1"));

        assert_eq!(url, "https://relay.example/api?foo=bar&x-id=1");
    }

    #[test]
    fn build_gemini_native_url_uses_origin_when_base_ends_with_v1beta() {
        let url = crate::proxy::gemini_url::build_gemini_native_url(
            "https://generativelanguage.googleapis.com/v1beta",
            "/v1beta/models/gemini-2.5-pro:generateContent",
        );

        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        );
    }

    #[test]
    fn build_gemini_native_url_uses_origin_when_base_already_contains_models_prefix() {
        let url = crate::proxy::gemini_url::build_gemini_native_url(
            "https://generativelanguage.googleapis.com/v1beta/models",
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
        );

        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn resolve_gemini_native_url_keeps_opaque_full_url_as_is() {
        let url = crate::proxy::gemini_url::resolve_gemini_native_url(
            "https://relay.example/custom/generate-content",
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            true,
        );

        assert_eq!(url, "https://relay.example/custom/generate-content?alt=sse");
    }

    #[test]
    fn force_identity_for_stream_flag_requests() {
        let headers = HeaderMap::new();

        assert!(should_force_identity_encoding(
            "/v1/responses",
            &json!({ "stream": true }),
            &headers
        ));
    }

    #[test]
    fn force_identity_for_gemini_stream_endpoints() {
        let headers = HeaderMap::new();

        assert!(should_force_identity_encoding(
            "/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
            &json!({ "model": "gemini-2.5-pro" }),
            &headers
        ));
    }

    #[test]
    fn streaming_request_detects_gemini_sse_without_body_stream_flag() {
        let headers = HeaderMap::new();

        assert!(is_streaming_request(
            "/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
            &json!({ "model": "gemini-2.5-pro" }),
            &headers
        ));
    }

    #[test]
    fn force_identity_for_sse_accept_header() {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));

        assert!(should_force_identity_encoding(
            "/v1/responses",
            &json!({ "model": "gpt-5" }),
            &headers
        ));
    }

    #[test]
    fn non_streaming_requests_allow_automatic_compression() {
        let headers = HeaderMap::new();

        assert!(!should_force_identity_encoding(
            "/v1/responses",
            &json!({ "model": "gpt-5" }),
            &headers
        ));
    }

    // ==================== Copilot 动态 endpoint 路由相关测试 ====================

    /// 验证 is_copilot 检测逻辑：通过 provider_type 判断
    #[test]
    fn copilot_detection_via_provider_type() {
        use crate::provider::{Provider, ProviderMeta};

        let provider = Provider {
            id: "test".to_string(),
            name: "Test Copilot".to_string(),
            settings_config: serde_json::json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(ProviderMeta {
                provider_type: Some("github_copilot".to_string()),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        };

        let is_copilot = provider
            .meta
            .as_ref()
            .and_then(|m| m.provider_type.as_deref())
            == Some("github_copilot");

        assert!(is_copilot, "应该通过 provider_type 检测为 Copilot");
    }

    /// 验证 is_copilot 检测逻辑：通过 base_url 判断
    #[test]
    fn copilot_detection_via_base_url() {
        let base_url = "https://api.githubcopilot.com";
        let is_copilot = base_url.contains("githubcopilot.com");
        assert!(is_copilot, "应该通过 base_url 检测为 Copilot");

        let non_copilot_url = "https://api.anthropic.com";
        let is_not_copilot = non_copilot_url.contains("githubcopilot.com");
        assert!(!is_not_copilot, "非 Copilot URL 不应被检测为 Copilot");
    }

    /// 验证企业版 endpoint（不包含 githubcopilot.com）场景下 is_copilot 仍然正确
    #[test]
    fn copilot_detection_for_enterprise_endpoint() {
        use crate::provider::{Provider, ProviderMeta};

        // 企业版场景：provider_type 是 github_copilot，但 base_url 可能是企业内部域名
        let provider = Provider {
            id: "enterprise".to_string(),
            name: "Enterprise Copilot".to_string(),
            settings_config: serde_json::json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(ProviderMeta {
                provider_type: Some("github_copilot".to_string()),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        };

        let enterprise_base_url = "https://copilot-api.corp.example.com";

        // is_copilot 应该通过 provider_type 检测成功，即使 base_url 不包含 githubcopilot.com
        let is_copilot = provider
            .meta
            .as_ref()
            .and_then(|m| m.provider_type.as_deref())
            == Some("github_copilot")
            || enterprise_base_url.contains("githubcopilot.com");

        assert!(
            is_copilot,
            "企业版 Copilot 应该通过 provider_type 被正确检测"
        );
    }

    /// 验证动态 endpoint 替换条件
    #[test]
    fn dynamic_endpoint_replacement_conditions() {
        // 条件：is_copilot && !is_full_url
        let test_cases = [
            (true, false, true, "Copilot + 非 full_url 应该替换"),
            (true, true, false, "Copilot + full_url 不应替换"),
            (false, false, false, "非 Copilot 不应替换"),
            (false, true, false, "非 Copilot + full_url 不应替换"),
        ];

        for (is_copilot, is_full_url, should_replace, desc) in test_cases {
            let will_replace = is_copilot && !is_full_url;
            assert_eq!(will_replace, should_replace, "{desc}");
        }
    }

    // ===== P3: forwarder 层 media 开关回归测试 =====
    // 验证 gate 在 forwarder 这一层的"接线"，而非 media_sanitizer 纯函数本身。

    fn forwarder_with_rectifier(config: RectifierConfig) -> RequestForwarder {
        let mut fwd = test_forwarder(Duration::from_secs(1), Duration::from_secs(1));
        fwd.rectifier_config = config;
        fwd
    }

    fn provider_with_settings(settings_config: Value) -> Provider {
        let mut p = test_provider_with_type(Some("anthropic"));
        p.settings_config = settings_config;
        p
    }

    fn body_with_image(model: &str) -> Value {
        json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "abc" } }
                ]
            }]
        })
    }

    fn body_with_codex_input_image(model: &str) -> Value {
        json!({
            "model": model,
            "input": [{
                "role": "user",
                "content": [
                    { "type": "input_image", "image_url": "data:image/png;base64,abc" }
                ]
            }]
        })
    }

    fn image_unsupported_error() -> ProxyError {
        ProxyError::UpstreamError {
            status: 400,
            body: Some(
                r#"{"error":{"message":"This model does not support image input"}}"#.to_string(),
            ),
        }
    }
    #[test]
    fn prevention_replaces_when_all_switches_on_and_model_in_heuristic_list() {
        let fwd = forwarder_with_rectifier(RectifierConfig::default());
        let provider = provider_with_settings(json!({}));
        let mut body = body_with_image("deepseek-v4-pro");

        let replaced = fwd.apply_media_prevention(&mut body, &provider);

        assert_eq!(replaced, 1, "默认全开 + 名单内模型应预替换");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
    }

    #[test]
    fn prevention_skipped_when_media_fallback_off() {
        // 关闭 request_media_fallback：即使名单命中也不预替换。
        let fwd = forwarder_with_rectifier(RectifierConfig {
            request_media_fallback: false,
            ..RectifierConfig::default()
        });
        let provider = provider_with_settings(json!({}));
        let mut body = body_with_image("deepseek-v4-pro");

        let replaced = fwd.apply_media_prevention(&mut body, &provider);

        assert_eq!(replaced, 0);
        assert_eq!(body["messages"][0]["content"][0]["type"], "image");
    }

    #[test]
    fn prevention_skipped_when_master_switch_off() {
        let fwd = forwarder_with_rectifier(RectifierConfig {
            enabled: false,
            ..RectifierConfig::default()
        });
        let provider = provider_with_settings(json!({}));
        let mut body = body_with_image("deepseek-v4-pro");

        assert_eq!(fwd.apply_media_prevention(&mut body, &provider), 0);
        assert_eq!(body["messages"][0]["content"][0]["type"], "image");
    }

    #[test]
    fn prevention_heuristic_off_skips_list_but_keeps_explicit_text_only() {
        // 关闭 request_media_heuristic：名单预测失效，但显式声明 text-only 仍预替换。
        let fwd = forwarder_with_rectifier(RectifierConfig {
            request_media_heuristic: false,
            ..RectifierConfig::default()
        });

        // (a) 名单内模型、无显式声明 → 不再预替换
        let bare_provider = provider_with_settings(json!({}));
        let mut list_body = body_with_image("deepseek-v4-pro");
        assert_eq!(
            fwd.apply_media_prevention(&mut list_body, &bare_provider),
            0,
            "heuristic 关闭后名单模型不应被预替换"
        );
        assert_eq!(list_body["messages"][0]["content"][0]["type"], "image");

        // (b) 显式声明 text-only → 仍预替换（声明驱动，不受 heuristic 开关影响）
        let declared_provider = provider_with_settings(json!({
            "models": [ { "id": "some-text-model", "input": ["text"] } ]
        }));
        let mut declared_body = body_with_image("some-text-model");
        assert_eq!(
            fwd.apply_media_prevention(&mut declared_body, &declared_provider),
            1,
            "显式 text-only 即使关闭 heuristic 也应预替换"
        );
        assert_eq!(declared_body["messages"][0]["content"][0]["type"], "text");
    }

    #[test]
    fn reactive_triggers_when_all_switches_on() {
        let fwd = forwarder_with_rectifier(RectifierConfig::default());
        let body = body_with_image("any-model");
        assert!(fwd.media_retry_should_trigger("Claude", false, &body, &image_unsupported_error()));
    }

    #[test]
    fn reactive_triggers_for_codex_image_url_deserialize_errors() {
        let fwd = forwarder_with_rectifier(RectifierConfig::default());
        let body = body_with_codex_input_image("deepseek-v4-flash");
        let error = ProxyError::UpstreamError {
            status: 400,
            body: Some(
                r#"{"error":{"message":"Failed to deserialize the JSON body into the target type: messages[11]: unknown variant image_url, expected text"}}"#
                    .to_string(),
            ),
        };

        assert!(fwd.media_retry_should_trigger("Codex", false, &body, &error));
    }

    #[test]
    fn reactive_skipped_when_media_fallback_off() {
        // 关闭 request_media_fallback：上游报图片错误也不触发兜底重试。
        let fwd = forwarder_with_rectifier(RectifierConfig {
            request_media_fallback: false,
            ..RectifierConfig::default()
        });
        let body = body_with_image("any-model");
        assert!(!fwd.media_retry_should_trigger(
            "Claude",
            false,
            &body,
            &image_unsupported_error()
        ));
    }

    #[test]
    fn reactive_skipped_when_master_switch_off() {
        let fwd = forwarder_with_rectifier(RectifierConfig {
            enabled: false,
            ..RectifierConfig::default()
        });
        let body = body_with_image("any-model");
        assert!(!fwd.media_retry_should_trigger(
            "Claude",
            false,
            &body,
            &image_unsupported_error()
        ));
    }

    #[test]
    fn reactive_unaffected_by_heuristic_switch() {
        // 关闭 request_media_heuristic 不影响反应式兜底——它是上游实测错误后的恢复，不是预测。
        let fwd = forwarder_with_rectifier(RectifierConfig {
            request_media_heuristic: false,
            ..RectifierConfig::default()
        });
        let body = body_with_image("any-model");
        assert!(fwd.media_retry_should_trigger("Claude", false, &body, &image_unsupported_error()));
    }
}

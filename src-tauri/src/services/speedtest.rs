use futures::future::join_all;
use reqwest::{Client, Url};
use serde::Serialize;
use std::time::Instant;

use crate::error::AppError;

const DEFAULT_TIMEOUT_SECS: u64 = 8;
const MAX_TIMEOUT_SECS: u64 = 30;
const MIN_TIMEOUT_SECS: u64 = 2;

/// 端点测速结果
#[derive(Debug, Clone, Serialize)]
pub struct EndpointLatency {
    pub url: String,
    pub latency: Option<u128>,
    pub status: Option<u16>,
    pub error: Option<String>,
}

/// 网络测速相关业务
pub struct SpeedtestService;

impl SpeedtestService {
    /// 测试一组端点的响应延迟。
    pub async fn test_endpoints(
        urls: Vec<String>,
        timeout_secs: Option<u64>,
    ) -> Result<Vec<EndpointLatency>, AppError> {
        if urls.is_empty() {
            return Ok(vec![]);
        }

        let mut results: Vec<Option<EndpointLatency>> = vec![None; urls.len()];
        let mut valid_targets = Vec::new();

        for (idx, raw_url) in urls.into_iter().enumerate() {
            let trimmed = raw_url.trim().to_string();

            if trimmed.is_empty() {
                results[idx] = Some(EndpointLatency {
                    url: raw_url,
                    latency: None,
                    status: None,
                    error: Some("URL 不能为空".to_string()),
                });
                continue;
            }

            match Url::parse(&trimmed) {
                Ok(parsed_url) => valid_targets.push((idx, trimmed, parsed_url)),
                Err(err) => {
                    results[idx] = Some(EndpointLatency {
                        url: trimmed,
                        latency: None,
                        status: None,
                        error: Some(format!("URL 无效: {err}")),
                    });
                }
            }
        }

        if valid_targets.is_empty() {
            return Ok(results.into_iter().flatten().collect::<Vec<_>>());
        }

        let timeout = Self::sanitize_timeout(timeout_secs);
        let (client, request_timeout) = Self::build_client(timeout)?;

        let tasks = valid_targets.into_iter().map(|(idx, trimmed, parsed_url)| {
            let client = client.clone();
            async move {
                // 构建健康检查 URL：优先使用 /v1/models 端点
                let health_check_url =
                    if parsed_url.path().ends_with("/v1") || parsed_url.path().ends_with("/v1/") {
                        parsed_url.join("models").ok()
                    } else {
                        parsed_url.join("v1/models").ok()
                    }
                    .unwrap_or_else(|| parsed_url.clone());

                // 先进行一次热身请求，忽略结果，仅用于复用连接/绕过首包惩罚。
                let _ = client
                    .get(health_check_url.clone())
                    .timeout(request_timeout)
                    .send()
                    .await;

                // 第二次请求开始计时，并将其作为结果返回。
                let start = Instant::now();
                let latency = match client
                    .get(health_check_url)
                    .timeout(request_timeout)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let status_code = resp.status().as_u16();
                        // 401 说明端点存在，只是需要认证，视为成功
                        let is_success = status_code < 500 || status_code == 401;

                        EndpointLatency {
                            url: trimmed,
                            latency: if is_success {
                                Some(start.elapsed().as_millis())
                            } else {
                                None
                            },
                            status: Some(status_code),
                            error: if is_success {
                                None
                            } else {
                                Some(format!("HTTP {}", status_code))
                            },
                        }
                    }
                    Err(err) => {
                        let status = err.status().map(|s| s.as_u16());
                        // 401 也视为成功（说明端点存在）
                        let is_auth_error = status == Some(401);

                        let error_message = if err.is_timeout() {
                            "请求超时".to_string()
                        } else if err.is_connect() {
                            "连接失败".to_string()
                        } else if is_auth_error {
                            // 401 不显示错误
                            String::new()
                        } else {
                            err.to_string()
                        };

                        EndpointLatency {
                            url: trimmed,
                            latency: if is_auth_error {
                                Some(start.elapsed().as_millis())
                            } else {
                                None
                            },
                            status,
                            error: if error_message.is_empty() {
                                None
                            } else {
                                Some(error_message)
                            },
                        }
                    }
                };

                (idx, latency)
            }
        });

        for (idx, latency) in join_all(tasks).await {
            results[idx] = Some(latency);
        }

        Ok(results.into_iter().flatten().collect::<Vec<_>>())
    }

    fn build_client(timeout_secs: u64) -> Result<(Client, std::time::Duration), AppError> {
        // 使用全局 HTTP 客户端（已包含代理配置）
        // 返回 timeout Duration 供请求级别使用
        let timeout = std::time::Duration::from_secs(timeout_secs);
        Ok((crate::proxy::http_client::get(), timeout))
    }

    fn sanitize_timeout(timeout_secs: Option<u64>) -> u64 {
        let secs = timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        secs.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_timeout_clamps_values() {
        assert_eq!(
            SpeedtestService::sanitize_timeout(Some(1)),
            MIN_TIMEOUT_SECS
        );
        assert_eq!(
            SpeedtestService::sanitize_timeout(Some(999)),
            MAX_TIMEOUT_SECS
        );
        assert_eq!(
            SpeedtestService::sanitize_timeout(Some(10)),
            10.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS)
        );
        assert_eq!(
            SpeedtestService::sanitize_timeout(None),
            DEFAULT_TIMEOUT_SECS
        );
    }

    #[test]
    fn test_endpoints_handles_empty_list() {
        let result =
            tauri::async_runtime::block_on(SpeedtestService::test_endpoints(Vec::new(), Some(5)))
                .expect("empty list should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_endpoints_reports_invalid_url() {
        let result = tauri::async_runtime::block_on(SpeedtestService::test_endpoints(
            vec!["not a url".into(), "".into()],
            None,
        ))
        .expect("invalid inputs should still succeed");

        assert_eq!(result.len(), 2);
        assert!(
            result[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .starts_with("URL 无效"),
            "invalid url should yield parse error"
        );
        assert_eq!(
            result[1].error.as_deref(),
            Some("URL 不能为空"),
            "empty url should report validation error"
        );
    }
}

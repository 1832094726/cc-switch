#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROXY_PORT="${CC_SWITCH_DEV_PROXY_PORT:-${CC_SWITCH_PORT:-15721}}"
RENDERER_PORT="${CC_SWITCH_DEV_RENDERER_PORT:-3000}"
FORCE_KILL_PROXY_PORT="${CC_SWITCH_DEV_FORCE_KILL_PORT:-0}"
FORCE_KILL_RENDERER_PORT="${CC_SWITCH_DEV_FORCE_KILL_RENDERER_PORT:-0}"

# 统一使用 src-tauri/target 作为编译缓存目录，与 tauri dev 一致，
# 避免预编译和 tauri dev 使用不同 target dir 导致全量重编译。
export CARGO_TARGET_DIR="${CC_SWITCH_DEV_CARGO_TARGET_DIR:-$ROOT_DIR/src-tauri/target}"

log() {
  printf '[cc-switch-dev] %s\n' "$*"
}

pid_is_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

wait_for_pid_exit() {
  local pid="$1"
  for attempt in 1 2 3 4 5; do
    if ! pid_is_running "$pid"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

cleanup_proxy_port() {
  local pids
  pids="$(lsof -tiTCP:"$PROXY_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  [[ -z "$pids" ]] && return 0

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$FORCE_KILL_PROXY_PORT" != "1" && "$command" != *"cc-switch"* ]]; then
      log "port $PROXY_PORT occupied by non-cc-switch process: pid=$pid cmd=$command"
      log "set CC_SWITCH_DEV_FORCE_KILL_PORT=1 to kill it anyway"
      exit 1
    fi
    log "stopping old proxy on port $PROXY_PORT: pid=$pid"
    kill "$pid" 2>/dev/null || true
    wait_for_pid_exit "$pid" || kill -9 "$pid" 2>/dev/null || true
  done <<< "$pids"
}

cleanup_renderer_port() {
  local pids
  pids="$(lsof -tiTCP:"$RENDERER_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  [[ -z "$pids" ]] && return 0

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$FORCE_KILL_RENDERER_PORT" != "1" ]]; then
      if [[ "$command" != *"vite"* && "$command" != *"node_modules/.bin/../vite/bin/vite.js"* ]]; then
        log "renderer port $RENDERER_PORT occupied by non-project process: pid=$pid cmd=$command"
        log "set CC_SWITCH_DEV_FORCE_KILL_RENDERER_PORT=1 to kill it anyway"
        exit 1
      fi
    fi
    log "stopping old renderer on port $RENDERER_PORT: pid=$pid"
    kill "$pid" 2>/dev/null || true
    wait_for_pid_exit "$pid" || kill -9 "$pid" 2>/dev/null || true
  done <<< "$pids"
}

cleanup_old_cc_switch_processes() {
  local current_pid="$$"
  local pids
  pids="$(
    ps -axo pid=,command= 2>/dev/null \
      | awk -v target="$CARGO_TARGET_DIR/debug/cc-switch" -v root="$ROOT_DIR" -v self="$current_pid" '
          $1 == self { next }
          {
            pid = $1
            $1 = ""
            sub(/^ +/, "", $0)
            cmd = $0
            if (cmd == target || cmd ~ target " " || cmd ~ root "/src-tauri/target/debug/cc-switch( |$)") {
              print pid
            }
          }
        '
  )"
  [[ -z "$pids" ]] && return 0

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    log "stopping old cc-switch backend: pid=$pid"
    kill "$pid" 2>/dev/null || true
    wait_for_pid_exit "$pid" || kill -9 "$pid" 2>/dev/null || true
  done <<< "$pids"
}

# ── 启动 ──────────────────────────────────────────────
log "cargo target dir: $CARGO_TARGET_DIR"

cleanup_old_cc_switch_processes
cleanup_proxy_port
cleanup_renderer_port

# 手动重载模式：--no-watch 禁用 Tauri 的 Rust 文件监听。
# 改完 Rust 代码后 Ctrl+C 停掉脚本、再重新运行即可。
# 前端 TS/CSS 仍然走 Vite HMR，不需要重启。
log "manual reload mode — no Rust auto-rebuild"
log "  Rust changes: Ctrl+C → re-run this script"
log "  Frontend (TS/CSS): auto via Vite HMR"
exec ./node_modules/.bin/tauri dev --no-watch \
  --config '{"identifier":"com.ccswitch.desktop.dev","build":{"beforeDevCommand":"npm run dev:renderer"}}'

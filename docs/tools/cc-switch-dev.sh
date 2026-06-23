#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROXY_PORT="${CC_SWITCH_DEV_PROXY_PORT:-${CC_SWITCH_PORT:-15721}}"
RENDERER_PORT="${CC_SWITCH_DEV_RENDERER_PORT:-3000}"
CLEAN_PROXY_PORT="${CC_SWITCH_DEV_CLEAN_PROXY_PORT:-1}"
CLEAN_RENDERER_PORT="${CC_SWITCH_DEV_CLEAN_RENDERER_PORT:-1}"
CLEAN_OLD_CC_SWITCH_PROCESSES="${CC_SWITCH_DEV_CLEAN_OLD_PROCESSES:-1}"
FORCE_KILL_PROXY_PORT="${CC_SWITCH_DEV_FORCE_KILL_PORT:-0}"
FORCE_KILL_RENDERER_PORT="${CC_SWITCH_DEV_FORCE_KILL_RENDERER_PORT:-0}"
PREBUILD_BACKEND="${CC_SWITCH_DEV_PREBUILD_BACKEND:-1}"
HOT_RELOAD_BACKEND="${CC_SWITCH_DEV_HOT_RELOAD_BACKEND:-1}"
HOT_RELOAD_INTERVAL="${CC_SWITCH_DEV_HOT_RELOAD_INTERVAL:-2}"
export CARGO_TARGET_DIR="${CC_SWITCH_DEV_CARGO_TARGET_DIR:-$HOME/.cache/cc-switch/cargo-target}"

log() {
  printf '[cc-switch-dev] %s\n' "$*"
}

pid_is_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

wait_for_pid_exit() {
  local pid="$1"
  local attempt

  for attempt in 1 2 3 4 5; do
    if ! pid_is_running "$pid"; then
      return 0
    fi
    sleep 0.2
  done

  return 1
}

cleanup_proxy_port() {
  if [[ "$CLEAN_PROXY_PORT" != "1" ]]; then
    return 0
  fi

  local pids
  pids="$(lsof -tiTCP:"$PROXY_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue

    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$FORCE_KILL_PROXY_PORT" != "1" && "$command" != *"cc-switch"* ]]; then
      log "port $PROXY_PORT is occupied by non-cc-switch process: pid=$pid command=$command"
      log "set CC_SWITCH_DEV_FORCE_KILL_PORT=1 to kill it anyway"
      exit 1
    fi

    log "stopping old proxy listener on port $PROXY_PORT: pid=$pid command=$command"
    kill "$pid" 2>/dev/null || true
    if ! wait_for_pid_exit "$pid"; then
      log "old process did not exit after SIGTERM, sending SIGKILL: pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"
}

cleanup_renderer_port() {
  if [[ "$CLEAN_RENDERER_PORT" != "1" ]]; then
    return 0
  fi

  local pids
  pids="$(lsof -tiTCP:"$RENDERER_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue

    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$FORCE_KILL_RENDERER_PORT" != "1" ]]; then
      if [[ "$command" != *"vite"* && "$command" != *"node_modules/.bin/../vite/bin/vite.js"* ]]; then
        log "renderer port $RENDERER_PORT is occupied by non-project process: pid=$pid command=$command"
        log "set CC_SWITCH_DEV_FORCE_KILL_RENDERER_PORT=1 to kill it anyway"
        exit 1
      fi
    fi

    log "stopping old renderer listener on port $RENDERER_PORT: pid=$pid command=$command"
    kill "$pid" 2>/dev/null || true
    if ! wait_for_pid_exit "$pid"; then
      log "old renderer process did not exit after SIGTERM, sending SIGKILL: pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"
}

cleanup_old_cc_switch_processes() {
  if [[ "$CLEAN_OLD_CC_SWITCH_PROCESSES" != "1" ]]; then
    return 0
  fi

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

  if [[ -z "$pids" ]]; then
    return 0
  fi

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    log "stopping old cc-switch backend: pid=$pid command=$command"
    kill "$pid" 2>/dev/null || true
    if ! wait_for_pid_exit "$pid"; then
      log "old backend did not exit after SIGTERM, sending SIGKILL: pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"
}

prebuild_backend() {
  if [[ "$PREBUILD_BACKEND" != "1" ]]; then
    return 0
  fi

  log "prebuilding backend from current source"
  cargo build --manifest-path "$ROOT_DIR/src-tauri/Cargo.toml"

  local binary="$CARGO_TARGET_DIR/debug/cc-switch"
  if [[ -x "$binary" ]]; then
    log "backend binary ready: $binary"
    log "backend binary mtime: $(stat -f '%Sm' "$binary" 2>/dev/null || stat -c '%y' "$binary" 2>/dev/null || true)"
  fi
}

backend_source_mtime() {
  (
    find "$ROOT_DIR/src-tauri/src" -type f \
      \( -name '*.rs' -o -name '*.toml' -o -name '*.json' \) -print 2>/dev/null
    find "$ROOT_DIR/src-tauri" -maxdepth 1 -type f \
      \( -name 'Cargo.toml' -o -name 'Cargo.lock' -o -name 'build.rs' -o -name 'tauri.conf*.json' \) -print 2>/dev/null
  ) | while IFS= read -r file; do
    stat -f '%m' "$file" 2>/dev/null || stat -c '%Y' "$file" 2>/dev/null || true
  done | sort -nr | head -n 1
}

stop_tauri_dev_child() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi

  if pid_is_running "$pid"; then
    log "stopping tauri dev process: pid=$pid"
    kill "$pid" 2>/dev/null || true
    wait_for_pid_exit "$pid" || kill -9 "$pid" 2>/dev/null || true
  fi

  cleanup_old_cc_switch_processes
  cleanup_proxy_port
  cleanup_renderer_port
}

run_tauri_dev_once() {
  ./node_modules/.bin/tauri dev --config '{"identifier":"com.ccswitch.desktop.dev","build":{"beforeDevCommand":"npm run dev:renderer"}}'
}

run_tauri_dev_hot_reload() {
  local last_mtime
  local next_mtime
  local child_pid=""

  last_mtime="$(backend_source_mtime)"
  log "backend hot reload enabled (interval=${HOT_RELOAD_INTERVAL}s)"

  trap 'stop_tauri_dev_child "$child_pid"; exit 130' INT TERM

  while true; do
    log "starting tauri dev"
    run_tauri_dev_once &
    child_pid="$!"

    while pid_is_running "$child_pid"; do
      sleep "$HOT_RELOAD_INTERVAL"
      next_mtime="$(backend_source_mtime)"
      if [[ -n "$next_mtime" && "$next_mtime" != "$last_mtime" ]]; then
        log "backend source changed, restarting tauri dev"
        last_mtime="$next_mtime"
        stop_tauri_dev_child "$child_pid"
        child_pid=""
        prebuild_backend
        break
      fi
    done

    if [[ -n "$child_pid" ]]; then
      set +e
      wait "$child_pid"
      local status="$?"
      set -e
      child_pid=""
      if [[ "$status" -ne 0 ]]; then
        log "tauri dev exited with status $status"
        return "$status"
      fi
      return 0
    fi
  done
}

mkdir -p "$CARGO_TARGET_DIR"
log "using cargo target dir: $CARGO_TARGET_DIR"

cleanup_old_cc_switch_processes
cleanup_proxy_port
cleanup_renderer_port

prebuild_backend

if [[ "$HOT_RELOAD_BACKEND" == "1" ]]; then
  run_tauri_dev_hot_reload
else
  exec ./node_modules/.bin/tauri dev --config '{"identifier":"com.ccswitch.desktop.dev","build":{"beforeDevCommand":"npm run dev:renderer"}}'
fi

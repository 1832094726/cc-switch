#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${DEVIN_APP_PATH:-/Applications/Devin.app}"
EXTENSION="${DEVIN_WINDSURF_EXTENSION:-$APP_PATH/Contents/Resources/app/extensions/windsurf/dist/extension.js}"
HOST="${CC_SWITCH_HOST:-127.0.0.1}"
PORT="${CC_SWITCH_PORT:-15721}"
API_URL="${CC_SWITCH_DEVIN_API_URL:-http://$HOST:$PORT/_route/api_server}"
INFERENCE_URL="${CC_SWITCH_DEVIN_INFERENCE_URL:-http://$HOST:$PORT/_route/inference}"

if [[ ! -f "$EXTENSION" ]]; then
  echo "Windsurf extension.js not found: $EXTENSION" >&2
  exit 1
fi

BACKUP="$EXTENSION.cc-switch.bak.$(date +%Y%m%d%H%M%S)"
if ! cp -X "$EXTENSION" "$BACKUP" 2>/dev/null; then
  cp "$EXTENSION" "$BACKUP"
fi

API_URL="$API_URL" INFERENCE_URL="$INFERENCE_URL" EXTENSION="$EXTENSION" node <<'NODE'
const fs = require("fs");

const file = process.env.EXTENSION;
const api = process.env.API_URL;
const inference = process.env.INFERENCE_URL;
let text = fs.readFileSync(file, "utf8");
let changes = 0;

function replaceAll(pattern, replacement) {
  const next = text.replace(pattern, replacement);
  if (next !== text) changes += 1;
  text = next;
}

replaceAll(
  /http:\/\/localhost:3000|http:\/\/127\.0\.0\.1:15721\/_route\/api_server/g,
  api,
);
replaceAll(
  /http:\/\/localhost:3001|http:\/\/127\.0\.0\.1:15721\/_route\/inference/g,
  inference,
);

if (!text.includes(api)) {
  replaceAll(
    /([A-Za-z_$][\w$]*)\.getApiServerUrlFromContext=([A-Za-z_$][\w$]*)=>\{if\(\(0,([A-Za-z_$][\w$]*)\.getConfig\)\(\3\.Config\.API_SERVER_URL\)!==([A-Za-z_$][\w$]*)\.DEFAULT_API_SERVER_URL\).*?return void 0===.*?\}/gs,
    (_match, ns, arg) =>
      `${ns}.getApiServerUrlFromContext=${arg}=>{return${JSON.stringify(api)}}`,
  );

  replaceAll(
    /async restart\(([A-Za-z_$][\w$]*)\)\{this\.apiServerUrl=\1,this\.inputs\.apiServerUrl=\1,/g,
    (_match, arg) =>
      `async restart(${arg}){${arg}=${JSON.stringify(api)},this.apiServerUrl=${arg},this.inputs.apiServerUrl=${arg},`,
  );
}

if (!text.includes(inference)) {
  replaceAll(
    /const ([A-Za-z_$][\w$]*)=\(0,([A-Za-z_$][\w$]*)\.getConfig\)\(\2\.Config\.INFERENCE_API_SERVER_URL\)/g,
    (_match, variable) => `const ${variable}=${JSON.stringify(inference)}`,
  );
}

if (changes === 0) {
  throw new Error(`No Devin/Windsurf route patch matched in ${file}`);
}

fs.writeFileSync(file, text);
NODE

if [[ "$(uname -s)" == "Darwin" && "${SKIP_CODESIGN:-0}" != "1" ]]; then
  codesign --force --deep --sign - "$APP_PATH" >/dev/null
fi

echo "Patched Devin Windsurf routes:"
echo "  api_server_url=$API_URL"
echo "  inference_api_server_url=$INFERENCE_URL"
echo "Backup: $BACKUP"

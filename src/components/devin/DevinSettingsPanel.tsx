import { useEffect, useMemo, useState } from "react";
import {
  ClipboardPaste,
  Loader2,
  Plus,
  Save,
  Settings2,
  Trash2,
} from "lucide-react";
import { parse as parseToml } from "smol-toml";
import { toast } from "sonner";
import { configApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { FullScreenPanel } from "@/components/common/FullScreenPanel";
import JsonEditor from "@/components/JsonEditor";

interface DevinSettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface VariableRow {
  id: string;
  key: string;
  value: string;
}

interface SmallModelSettings {
  baseUrl: string;
  apiKey: string;
  model: string;
  endpoint: string;
  requestModels: string;
}

const DEFAULT_SMALL_MODEL_SETTINGS: SmallModelSettings = {
  baseUrl: "https://api.siliconflow.cn",
  apiKey: "",
  model: "deepseek-ai/DeepSeek-V3.2",
  endpoint: "/v1/chat/completions",
  requestModels:
    "MODEL_GPT_5_NANO, MODEL_GOOGLE_GEMINI_2_5_FLASH, MODEL_CHAT_GPT_4_1_MINI_2025_04_14",
};

const DEFAULT_DEVIN_SNIPPET = `# Devin common variables
# Saved here only. It will not reload Devin or rewrite providers automatically.

# Example:
# pt_key = "BJ.xxxxx"
# joycode_cookie = """pt_key=...; pt_pin=..."""
# model_reasoning_effort = "high"
# approval_policy = "never"

# Or use explicit headers:
# [joycode.headers]
# pt_key = "BJ.xxxxx"
# cookie = """pt_key=...; pt_pin=..."""

# Devin small/compression models are independent from the selected main provider.
# They are used when Devin requests non-primary helper/compression models.
# [small_models]
# enabled = true
# base_url = "https://api.siliconflow.cn"
# api_key = "sk-..."
# model = "deepseek-ai/DeepSeek-V3.2"
# endpoint = "/v1/chat/completions"
# thinking_enabled = false
# request_models = ["MODEL_GPT_5_NANO", "MODEL_GOOGLE_GEMINI_2_5_FLASH", "MODEL_CHAT_GPT_4_1_MINI_2025_04_14"]
`;

const VARIABLE_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_.-]*$/;

const quoteTomlString = (value: string) =>
  `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;

const quoteTomlMultilineString = (value: string) =>
  `"""${value.replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"')}"""`;

const upsertTopLevelTomlString = (
  source: string,
  key: string,
  value: string,
) => {
  const line = `${key} = ${quoteTomlMultilineString(value)}`;
  const pattern = new RegExp(
    `(^|\\n)\\s*${key}\\s*=\\s*(?:"""[\\s\\S]*?"""|"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|[^\\n]*)`,
    "m",
  );

  if (pattern.test(source)) {
    return source.replace(pattern, `$1${line}`);
  }

  const base = source.trimEnd();
  return `${base}${base ? "\n\n" : ""}${line}\n`;
};

const buildVariableLines = (rows: VariableRow[]) =>
  rows
    .map((row) => ({
      key: row.key.trim(),
      value: row.value,
    }))
    .filter((row) => row.key)
    .map((row) => `${row.key} = ${quoteTomlString(row.value)}`)
    .join("\n");

const tomlFieldToString = (value: unknown) =>
  typeof value === "string" ? value : "";

const readSmallModelSettings = (source: string): SmallModelSettings => {
  try {
    const parsed = parseToml(source) as Record<string, unknown>;
    const table =
      parsed.small_models && typeof parsed.small_models === "object"
        ? (parsed.small_models as Record<string, unknown>)
        : {};
    return {
      baseUrl:
        tomlFieldToString(table.base_url) ||
        tomlFieldToString(table.baseUrl) ||
        DEFAULT_SMALL_MODEL_SETTINGS.baseUrl,
      apiKey:
        tomlFieldToString(table.api_key) ||
        tomlFieldToString(table.apiKey) ||
        DEFAULT_SMALL_MODEL_SETTINGS.apiKey,
      model:
        tomlFieldToString(table.model) ||
        tomlFieldToString(table.upstream_model) ||
        tomlFieldToString(table.upstreamModel) ||
        DEFAULT_SMALL_MODEL_SETTINGS.model,
      endpoint:
        tomlFieldToString(table.endpoint) ||
        DEFAULT_SMALL_MODEL_SETTINGS.endpoint,
      requestModels: Array.isArray(table.request_models)
        ? table.request_models.filter((item) => typeof item === "string").join(", ")
        : Array.isArray(table.requestModels)
          ? table.requestModels.filter((item) => typeof item === "string").join(", ")
          : tomlFieldToString(table.request_models) ||
            tomlFieldToString(table.requestModels) ||
            tomlFieldToString(table.aliases) ||
            DEFAULT_SMALL_MODEL_SETTINGS.requestModels,
    };
  } catch {
    return DEFAULT_SMALL_MODEL_SETTINGS;
  }
};

const upsertSmallModelsToml = (
  source: string,
  settings: SmallModelSettings,
) => {
  const block = [
    "[small_models]",
    "enabled = true",
    `base_url = ${quoteTomlString(settings.baseUrl.trim())}`,
    `api_key = ${quoteTomlString(settings.apiKey.trim())}`,
    `model = ${quoteTomlString(settings.model.trim())}`,
    `endpoint = ${quoteTomlString(settings.endpoint.trim())}`,
    "thinking_enabled = false",
    `request_models = [${settings.requestModels
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .map(quoteTomlString)
      .join(", ")}]`,
  ].join("\n");
  const pattern = /(^|\n)\s*\[small_models\][\s\S]*?(?=\n\s*\[[^\]]+\]|\s*$)/m;

  if (pattern.test(source)) {
    return source.replace(pattern, `$1${block}\n`);
  }

  const base = source.trimEnd();
  return `${base}${base ? "\n\n" : ""}${block}\n`;
};

export function DevinSettingsPanel({
  open,
  onOpenChange,
}: DevinSettingsPanelProps) {
  const [snippet, setSnippet] = useState(DEFAULT_DEVIN_SNIPPET);
  const [initialSnippet, setInitialSnippet] = useState(DEFAULT_DEVIN_SNIPPET);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [rows, setRows] = useState<VariableRow[]>([]);
  const [joycodeCookie, setJoycodeCookie] = useState("");
  const [smallModels, setSmallModels] = useState<SmallModelSettings>(
    DEFAULT_SMALL_MODEL_SETTINGS,
  );

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    setIsLoading(true);
    configApi
      .getCommonConfigSnippet("devin")
      .then((value) => {
        if (cancelled) return;
        const next = value?.trim() ? value : DEFAULT_DEVIN_SNIPPET;
        setSnippet(next);
        setInitialSnippet(next);
        setRows([]);
        setJoycodeCookie("");
        setSmallModels(readSmallModelSettings(next));
      })
      .catch((error) => {
        if (cancelled) return;
        toast.error(`读取 Devin 设置失败: ${String(error)}`);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open]);

  const validationError = useMemo(() => {
    if (!snippet.trim()) return "";
    try {
      parseToml(snippet);
      return "";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  }, [snippet]);

  const variableError = useMemo(() => {
    for (const row of rows) {
      const key = row.key.trim();
      if (!key) continue;
      if (!VARIABLE_KEY_PATTERN.test(key)) {
        return `变量名 "${key}" 只能包含字母、数字、下划线、点和短横线，且不能以数字开头`;
      }
    }
    return "";
  }, [rows]);

  const isDirty = snippet !== initialSnippet || rows.length > 0;

  const handleAddRow = () => {
    setRows((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        key: "",
        value: "",
      },
    ]);
  };

  const handleApplyRows = () => {
    if (variableError) return;
    const lines = buildVariableLines(rows);
    if (!lines) return;
    setSnippet((prev) => {
      const base = prev.trimEnd();
      return `${base}${base ? "\n\n" : ""}${lines}\n`;
    });
    setRows([]);
  };

  const handleApplyJoycodeCookie = () => {
    const value = joycodeCookie.trim();
    if (!value) return;
    setSnippet((prev) =>
      upsertTopLevelTomlString(prev, "joycode_cookie", value),
    );
    setJoycodeCookie("");
    setSmallModels(DEFAULT_SMALL_MODEL_SETTINGS);
  };

  const handleApplySmallModels = () => {
    setSnippet((prev) => upsertSmallModelsToml(prev, smallModels));
  };

  const handleSave = async () => {
    if (variableError || validationError) return;

    setIsSaving(true);
    try {
      await configApi.setCommonConfigSnippet("devin", snippet);
      setInitialSnippet(snippet);
      toast.success("Devin 设置已保存");
      onOpenChange(false);
    } catch (error) {
      toast.error(`保存 Devin 设置失败: ${String(error)}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleClose = () => {
    setSnippet(initialSnippet);
    setRows([]);
    setJoycodeCookie("");
    onOpenChange(false);
  };

  return (
    <FullScreenPanel
      isOpen={open}
      title="Devin 设置"
      onClose={handleClose}
      footer={
        <>
          <Button type="button" variant="outline" onClick={handleClose}>
            取消
          </Button>
          <Button
            type="button"
            onClick={handleSave}
            disabled={isSaving || isLoading || Boolean(validationError)}
            className="gap-2"
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            保存
          </Button>
        </>
      }
    >
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
        <div className="rounded-lg border border-border-default bg-card p-4">
          <div className="mb-4 flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-muted-foreground" />
            <div>
              <h3 className="text-sm font-medium">公共变量</h3>
              <p className="text-xs text-muted-foreground">
                这些内容仅保存为 Devin
                通用配置片段，不会自动重载客户端或改写供应商。JoyCode 的 pt_key
                会在健康检查和代理请求时作为请求头传给本地桥接。
              </p>
            </div>
          </div>

          <div className="mb-4 space-y-2 rounded-md border border-border-default bg-background/40 p-3">
            <Label htmlFor="joycode-cookie">JoyCode Cookie</Label>
            <Textarea
              id="joycode-cookie"
              value={joycodeCookie}
              onChange={(event) => setJoycodeCookie(event.target.value)}
              placeholder="pt_key=...; pt_pin=...; ..."
              rows={4}
              className="font-mono text-xs"
            />
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">
                保存为 joycode_cookie，运行时会转成 Cookie 头并提取 pt_key。
              </p>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleApplyJoycodeCookie}
                disabled={!joycodeCookie.trim()}
                className="gap-2"
              >
                <ClipboardPaste className="h-4 w-4" />
                写入 JoyCode Cookie
              </Button>
            </div>
          </div>

          <div className="mb-4 space-y-3 rounded-md border border-border-default bg-background/40 p-3">
            <div>
              <Label>小模型路由</Label>
              <p className="mt-1 text-xs text-muted-foreground">
                专门接管 Devin 的非主模型请求，不占用当前供应商的 Base URL。
              </p>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="small-model-base-url">Base URL</Label>
                <Input
                  id="small-model-base-url"
                  value={smallModels.baseUrl}
                  onChange={(event) =>
                    setSmallModels((prev) => ({
                      ...prev,
                      baseUrl: event.target.value,
                    }))
                  }
                  placeholder="https://api.siliconflow.cn"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="small-model-endpoint">Endpoint</Label>
                <Input
                  id="small-model-endpoint"
                  value={smallModels.endpoint}
                  onChange={(event) =>
                    setSmallModels((prev) => ({
                      ...prev,
                      endpoint: event.target.value,
                    }))
                  }
                  placeholder="/v1/chat/completions"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="small-model-name">模型</Label>
                <Input
                  id="small-model-name"
                  value={smallModels.model}
                  onChange={(event) =>
                    setSmallModels((prev) => ({
                      ...prev,
                      model: event.target.value,
                    }))
                  }
                  placeholder="deepseek-ai/DeepSeek-V3.2"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="small-model-api-key">API Key</Label>
                <Input
                  id="small-model-api-key"
                  type="password"
                  value={smallModels.apiKey}
                  onChange={(event) =>
                    setSmallModels((prev) => ({
                      ...prev,
                      apiKey: event.target.value,
                    }))
                  }
                  placeholder="sk-..."
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="small-model-request-models">触发模型</Label>
              <Textarea
                id="small-model-request-models"
                value={smallModels.requestModels}
                onChange={(event) =>
                  setSmallModels((prev) => ({
                    ...prev,
                    requestModels: event.target.value,
                  }))
                }
                placeholder="MODEL_GPT_5_NANO, MODEL_GOOGLE_GEMINI_2_5_FLASH, MODEL_CHAT_GPT_4_1_MINI_2025_04_14"
                rows={2}
                className="font-mono text-xs"
              />
              <p className="text-xs text-muted-foreground">
                日志里出现的这些请求模型会被强制路由到上面的小模型供应商。
              </p>
            </div>
            <div className="flex justify-end">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleApplySmallModels}
                disabled={
                  !smallModels.baseUrl.trim() ||
                  !smallModels.model.trim() ||
                  !smallModels.endpoint.trim() ||
                  !smallModels.requestModels.trim()
                }
              >
                写入小模型路由
              </Button>
            </div>
          </div>

          <div className="space-y-3">
            {rows.map((row, index) => (
              <div
                key={row.id}
                className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] gap-2"
              >
                <Input
                  value={row.key}
                  onChange={(event) =>
                    setRows((prev) =>
                      prev.map((item, itemIndex) =>
                        itemIndex === index
                          ? { ...item, key: event.target.value }
                          : item,
                      ),
                    )
                  }
                  placeholder="变量名，例如 pt_key"
                />
                <Input
                  value={row.value}
                  onChange={(event) =>
                    setRows((prev) =>
                      prev.map((item, itemIndex) =>
                        itemIndex === index
                          ? { ...item, value: event.target.value }
                          : item,
                      ),
                    )
                  }
                  placeholder="变量值，例如 BJ.xxxxx"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() =>
                    setRows((prev) =>
                      prev.filter((_, itemIndex) => itemIndex !== index),
                    )
                  }
                  title="删除变量"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}

            {variableError && (
              <p className="text-xs text-red-500">{variableError}</p>
            )}

            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleAddRow}
                className="gap-2"
              >
                <Plus className="h-4 w-4" />
                添加变量
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleApplyRows}
                disabled={!rows.length || Boolean(variableError)}
              >
                写入 TOML
              </Button>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <Label>原始 TOML</Label>
          {isLoading ? (
            <div className="flex h-64 items-center justify-center rounded-lg border border-border-default">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <JsonEditor
              value={snippet}
              onChange={setSnippet}
              placeholder={DEFAULT_DEVIN_SNIPPET}
              rows={18}
              language="javascript"
              showValidation={false}
            />
          )}
          {validationError ? (
            <p className="text-xs text-red-500">{validationError}</p>
          ) : (
            <p className="text-xs text-muted-foreground">
              保存后，新建或编辑 Devin
              供应商时可手动启用通用配置；不会自动应用到正在使用的客户端。
              JoyCode 变量会在代理请求时运行时读取，不需要重载 Devin。
              {isDirty ? " 当前有未保存更改。" : ""}
            </p>
          )}
        </div>
      </div>
    </FullScreenPanel>
  );
}

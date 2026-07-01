import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { FormLabel } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { toast } from "sonner";
import {
  ChevronDown,
  ChevronRight,
  Download,
  ExternalLink,
  Loader2,
  Plus,
  Trash2,
} from "lucide-react";
import EndpointSpeedTest from "./EndpointSpeedTest";
import { ApiKeySection, EndpointField, ModelDropdown } from "./shared";
import {
  fetchJoycodeModelsForConfig,
  fetchModelsForConfig,
  showFetchModelsError,
  type FetchedModel,
} from "@/lib/api/model-fetch";
import { settingsApi } from "@/lib/api";
import { CustomUserAgentField } from "./CustomUserAgentField";
import { DEVIN_WINDSURF_MODEL_OPTIONS } from "@/config/devinProviderPresets";
import { LocalProxyRequestOverridesField } from "./LocalProxyRequestOverridesField";
import { cn } from "@/lib/utils";
import type {
  CodexApiFormat,
  CodexCatalogModel,
  CodexChatReasoning,
  ProviderCategory,
} from "@/types";

interface EndpointCandidate {
  url: string;
}

interface CodexFormFieldsProps {
  appId?: "codex" | "devin";
  providerId?: string;
  // API Key
  codexApiKey: string;
  onApiKeyChange: (key: string) => void;
  category?: ProviderCategory;
  shouldShowApiKeyLink: boolean;
  websiteUrl: string;
  isPartner?: boolean;
  partnerPromotionKey?: string;

  // Base URL
  shouldShowSpeedTest: boolean;
  codexBaseUrl: string;
  onBaseUrlChange: (url: string) => void;
  isFullUrl: boolean;
  onFullUrlChange: (value: boolean) => void;
  isEndpointModalOpen: boolean;
  onEndpointModalToggle: (open: boolean) => void;
  onCustomEndpointsChange?: (endpoints: string[]) => void;
  autoSelect: boolean;
  onAutoSelectChange: (checked: boolean) => void;

  // API Format
  // Note: wire_api is always "responses" for Codex; apiFormat controls proxy-layer conversion
  apiFormat: CodexApiFormat;
  onApiFormatChange: (format: CodexApiFormat) => void;
  codexChatReasoning?: CodexChatReasoning;
  onCodexChatReasoningChange?: (value: CodexChatReasoning) => void;

  // Model Catalog
  catalogModels?: CodexCatalogModel[];
  onCatalogModelsChange?: (models: CodexCatalogModel[]) => void;

  // Speed Test Endpoints
  speedTestEndpoints: EndpointCandidate[];

  // Local proxy User-Agent override
  customUserAgent: string;
  onCustomUserAgentChange: (value: string) => void;
  localProxyHeadersOverride: string;
  onLocalProxyHeadersOverrideChange: (value: string) => void;
  localProxyBodyOverride: string;
  onLocalProxyBodyOverrideChange: (value: string) => void;
}

type CodexCatalogRow = CodexCatalogModel & { rowId: string };

const DEVIN_CUSTOM_MODEL_VALUE = "__cc_switch_devin_custom_model__";
const DEVIN_DEFAULT_AUTH_HEADER: NonNullable<CodexCatalogModel["authHeader"]> =
  "bearer";
const JOYCODE_LOGIN_URL =
  "http://joycoder.jd.com?login=1&ideAppName=vscode&fromIde=joycode-plugin&redirect=0";
const JOYCODE_DEVIN_ROUTE_DEFAULTS: Record<
  string,
  Pick<
    CodexCatalogModel,
    "upstreamModel" | "endpoint" | "authHeader" | "thinkingEnabled"
  >
> = {
  "swe-1-6-slow": {
    upstreamModel: "GPT 5.3-codex",
    endpoint: "/v1/responses",
    authHeader: "bearer",
  },
  MODEL_CLAUDE_4_SONNET_BYOK: {
    upstreamModel: "Claude-Sonnet-4.6-hq",
    endpoint: "/v1/chat/completions",
    authHeader: "bearer",
  },
  MODEL_CLAUDE_4_OPUS_THINKING_BYOK: {
    upstreamModel: "Claude-Opus-4.6-hq",
    endpoint: "/v1/chat/completions",
    authHeader: "bearer",
  },
  MODEL_CLAUDE_4_OPUS_BYOK: {
    upstreamModel: "Claude-Opus-4.6-hq",
    endpoint: "/v1/chat/completions",
    authHeader: "bearer",
  },
};

function createCatalogRow(seed?: Partial<CodexCatalogModel>): CodexCatalogRow {
  const firstRoute = seed?.routes?.[0];
  return {
    rowId: crypto.randomUUID(),
    model: seed?.model ?? "",
    displayName: seed?.displayName ?? "",
    contextWindow: seed?.contextWindow ?? "",
    upstreamModel: seed?.upstreamModel ?? "",
    provider: seed?.provider,
    endpoint: seed?.endpoint,
    baseUrl: seed?.baseUrl ?? firstRoute?.baseUrl ?? "",
    apiKey: seed?.apiKey ?? firstRoute?.apiKey ?? "",
    routeName: seed?.routeName ?? firstRoute?.name ?? "",
    authHeader: seed?.authHeader ?? firstRoute?.authHeader,
    headers: seed?.headers ?? firstRoute?.headers,
    responsesMode: seed?.responsesMode ?? firstRoute?.responsesMode,
    responsesCodexCompat:
      seed?.responsesCodexCompat ?? firstRoute?.responsesCodexCompat,
    responsesFastMode: seed?.responsesFastMode ?? firstRoute?.responsesFastMode,
    thinkingEnabled: seed?.thinkingEnabled ?? firstRoute?.thinkingEnabled,
    // Carry native-profile overrides verbatim (not user-editable in the row UI,
    // but must survive load->save so the official catalog fidelity is kept).
    ...(seed?.supportsParallelToolCalls !== undefined
      ? { supportsParallelToolCalls: seed.supportsParallelToolCalls }
      : {}),
    ...(seed?.inputModalities ? { inputModalities: seed.inputModalities } : {}),
    ...(seed?.baseInstructions
      ? { baseInstructions: seed.baseInstructions }
      : {}),
  };
}

// Compares rows (with rowId) to incoming models (without) by data fields only,
// so both sync effects can use the same equality definition. Hidden native-profile
// fields are included so switching between providers with identical visible fields
// but different base_instructions / tools / modalities still rebuilds the rows.
function catalogRowsMatchModels(
  rows: CodexCatalogModel[],
  models: CodexCatalogModel[],
): boolean {
  if (rows.length !== models.length) return false;
  return rows.every((row, i) => {
    const incoming = models[i];
    return (
      row.model === (incoming.model ?? "") &&
      (row.displayName ?? "") === (incoming.displayName ?? "") &&
      String(row.contextWindow ?? "") ===
        String(incoming.contextWindow ?? "") &&
      (row.upstreamModel ?? "") === (incoming.upstreamModel ?? "") &&
      (row.provider ?? "") === (incoming.provider ?? "") &&
      (row.endpoint ?? "") === (incoming.endpoint ?? "") &&
      (row.baseUrl ?? "") ===
        (incoming.baseUrl ?? incoming.routes?.[0]?.baseUrl ?? "") &&
      (row.apiKey ?? "") ===
        (incoming.apiKey ?? incoming.routes?.[0]?.apiKey ?? "") &&
      (row.routeName ?? "") ===
        (incoming.routeName ?? incoming.routes?.[0]?.name ?? "") &&
      (row.authHeader ?? "") ===
        (incoming.authHeader ?? incoming.routes?.[0]?.authHeader ?? "") &&
      JSON.stringify(row.headers ?? {}) ===
        JSON.stringify(
          incoming.headers ?? incoming.routes?.[0]?.headers ?? {},
        ) &&
      (row.responsesMode ?? "") ===
        (incoming.responsesMode ?? incoming.routes?.[0]?.responsesMode ?? "") &&
      (row.responsesCodexCompat ?? undefined) ===
        (incoming.responsesCodexCompat ??
          incoming.routes?.[0]?.responsesCodexCompat ??
          undefined) &&
      (row.responsesFastMode ?? undefined) ===
        (incoming.responsesFastMode ??
          incoming.routes?.[0]?.responsesFastMode ??
          undefined) &&
      (row.thinkingEnabled ?? undefined) ===
        (incoming.thinkingEnabled ??
          incoming.routes?.[0]?.thinkingEnabled ??
          undefined) &&
      (row.supportsParallelToolCalls ?? null) ===
        (incoming.supportsParallelToolCalls ?? null) &&
      (row.baseInstructions ?? "") === (incoming.baseInstructions ?? "") &&
      JSON.stringify(row.inputModalities ?? []) ===
        JSON.stringify(incoming.inputModalities ?? [])
    );
  });
}

export function CodexFormFields({
  appId = "codex",
  providerId,
  codexApiKey,
  onApiKeyChange,
  category,
  shouldShowApiKeyLink,
  websiteUrl,
  isPartner,
  partnerPromotionKey,
  shouldShowSpeedTest,
  codexBaseUrl,
  onBaseUrlChange,
  isFullUrl,
  onFullUrlChange,
  isEndpointModalOpen,
  onEndpointModalToggle,
  onCustomEndpointsChange,
  autoSelect,
  onAutoSelectChange,
  apiFormat,
  onApiFormatChange,
  codexChatReasoning = {},
  onCodexChatReasoningChange,
  catalogModels = [],
  onCatalogModelsChange,
  speedTestEndpoints,
  customUserAgent,
  onCustomUserAgentChange,
  localProxyHeadersOverride,
  onLocalProxyHeadersOverrideChange,
  localProxyBodyOverride,
  onLocalProxyBodyOverrideChange,
}: CodexFormFieldsProps) {
  const { t } = useTranslation();

  const [fetchedModels, setFetchedModels] = useState<FetchedModel[]>([]);
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const isDevin = appId === "devin";
  const isJoyCodeProvider =
    (providerId ?? "").toLowerCase().includes("joycode") ||
    codexBaseUrl.toLowerCase().includes("joycode-api.jd.com");
  const isJoyCodeDevinProvider =
    isDevin &&
    (isJoyCodeProvider || /(?:127\.0\.0\.1|localhost):8081/.test(codexBaseUrl));
  // 思考能力随 Chat 格式显示（仅 Chat Completions 转换路径用得上）；模型映射常驻
  //（填了才生成 catalog）。两者都已与「路由接管」概念解耦。
  const isChatFormat = apiFormat === "openai_chat";
  const canEditCatalog = Boolean(onCatalogModelsChange);
  const canEditReasoning = Boolean(onCodexChatReasoningChange);
  const supportsThinking =
    codexChatReasoning.supportsThinking === true ||
    codexChatReasoning.supportsEffort === true;
  const supportsEffort = codexChatReasoning.supportsEffort === true;

  // 高级区在有任何可见配置时自动展开（仅折叠→展开，不会自动折叠）：自定义 UA /
  // 请求覆盖 / 已填模型映射 / 已配置思考能力。
  const hasRequestOverrides = Boolean(
    localProxyHeadersOverride.trim() || localProxyBodyOverride.trim(),
  );
  const hasAnyAdvancedValue =
    !!customUserAgent ||
    hasRequestOverrides ||
    apiFormat !== "openai_responses" ||
    catalogModels.length > 0 ||
    supportsThinking ||
    supportsEffort;
  const [advancedExpanded, setAdvancedExpanded] = useState(hasAnyAdvancedValue);

  // 预设/编辑加载填充高级值后自动展开（仅从折叠→展开，不会自动折叠）
  useEffect(() => {
    if (hasAnyAdvancedValue) {
      setAdvancedExpanded(true);
    }
  }, [hasAnyAdvancedValue]);

  const [catalogRows, setCatalogRows] = useState<CodexCatalogRow[]>(() =>
    catalogModels.map((m) => createCatalogRow(m)),
  );

  // 记录上次发送给父组件的数据，避免重复触发
  const lastSentModelsRef = useRef<CodexCatalogModel[]>(catalogModels);

  // 父 → 子：仅当 prop 数据真的变化（预设切换 / 编辑加载）时才重建 rowId；
  // 同 shape 时保留现有 rowId，避免编辑过程中焦点丢失。
  useEffect(() => {
    setCatalogRows((current) => {
      if (catalogRowsMatchModels(current, catalogModels)) return current;
      return catalogModels.map((m) => createCatalogRow(m));
    });
    // 同步更新 ref，避免父组件传入新数据时子→父 effect 误判为本地修改
    lastSentModelsRef.current = catalogModels;
  }, [catalogModels]);

  // 子 → 父：rowId 是视图层概念，不应进入持久化数据；剥离后再回传。
  // 注意：依赖数组不包含 catalogModels，避免父→子更新触发子→父回调形成循环。
  useEffect(() => {
    if (!onCatalogModelsChange) return;
    const next: CodexCatalogModel[] = catalogRows.map(
      ({
        rowId: _rowId,
        baseUrl,
        apiKey,
        routeName,
        authHeader,
        endpoint,
        provider,
        ...rest
      }) => {
        if (!isDevin) return rest;
        const normalizedEndpoint = endpoint ?? "/v1/responses";
        const routeBaseUrl = (
          isDevin ? codexBaseUrl || baseUrl : baseUrl
        )?.trim();
        const routeApiKey = (isDevin ? codexApiKey || apiKey : apiKey)?.trim();
        const routeAuthHeader =
          authHeader ??
          (normalizedEndpoint === "/v1/messages" ? "x-api-key" : "bearer");
        const responsesFields =
          normalizedEndpoint === "/v1/responses"
            ? {
                ...(rest.responsesMode
                  ? { responsesMode: rest.responsesMode }
                  : {}),
                ...(rest.responsesCodexCompat !== undefined
                  ? { responsesCodexCompat: rest.responsesCodexCompat }
                  : {}),
                ...(rest.responsesFastMode !== undefined
                  ? { responsesFastMode: rest.responsesFastMode }
                  : {}),
              }
            : {};
        const routeHeaders =
          rest.headers && Object.keys(rest.headers).length > 0
            ? rest.headers
            : undefined;
        return {
          ...rest,
          provider:
            provider ??
            (normalizedEndpoint === "/v1/messages" ? "anthropic" : "openai"),
          endpoint: normalizedEndpoint,
          ...(routeBaseUrl ? { baseUrl: routeBaseUrl } : {}),
          ...(routeApiKey ? { apiKey: routeApiKey } : {}),
          ...(routeAuthHeader ? { authHeader: routeAuthHeader } : {}),
          ...responsesFields,
          routes:
            routeBaseUrl || routeApiKey
              ? [
                  {
                    name: (routeName ?? "").trim() || "primary",
                    baseUrl: routeBaseUrl,
                    apiKey: routeApiKey,
                    enabled: true,
                    priority: 10,
                    authHeader: routeAuthHeader,
                    ...(routeHeaders ? { headers: routeHeaders } : {}),
                    ...responsesFields,
                    ...(rest.thinkingEnabled !== undefined
                      ? { thinkingEnabled: rest.thinkingEnabled }
                      : {}),
                  },
                ]
              : undefined,
        };
      },
    );
    // 只有当数据真的变化时才通知父组件
    if (catalogRowsMatchModels(next, lastSentModelsRef.current)) return;
    lastSentModelsRef.current = next;
    onCatalogModelsChange(next);
  }, [catalogRows, codexApiKey, codexBaseUrl, isDevin, onCatalogModelsChange]);

  const handleReasoningThinkingChange = useCallback(
    (checked: boolean) => {
      if (!onCodexChatReasoningChange) return;
      onCodexChatReasoningChange({
        ...codexChatReasoning,
        supportsThinking: checked,
        supportsEffort: checked ? codexChatReasoning.supportsEffort : false,
      });
    },
    [codexChatReasoning, onCodexChatReasoningChange],
  );

  const handleReasoningEffortChange = useCallback(
    (checked: boolean) => {
      if (!onCodexChatReasoningChange) return;
      onCodexChatReasoningChange({
        ...codexChatReasoning,
        supportsThinking: checked ? true : codexChatReasoning.supportsThinking,
        supportsEffort: checked,
        effortParam: checked
          ? (codexChatReasoning.effortParam ?? "reasoning_effort")
          : "none",
      });
    },
    [codexChatReasoning, onCodexChatReasoningChange],
  );

  const handleJoyCodeLogin = useCallback(() => {
    settingsApi
      .openExternal(JOYCODE_LOGIN_URL)
      .catch((err) => {
        console.warn("[JoyCode] Failed to open login URL:", err);
        toast.error(
          t("providerForm.joycodeLoginFailed", {
            defaultValue: "打开 JoyCode 登录页失败",
          }),
        );
      });
  }, [t]);

  const mergeFetchedModelsIntoCatalog = useCallback(
    (models: FetchedModel[]) => {
      if (!onCatalogModelsChange || !isJoyCodeProvider || models.length === 0) {
        return;
      }
      setCatalogRows((current) => {
        const seen = new Set(current.map((row) => row.model));
        const additions = models
          .filter((model) => model.id && !seen.has(model.id))
          .map((model) =>
            createCatalogRow({
              model: model.id,
              displayName: model.id,
              contextWindow: 256000,
            }),
          );
        return additions.length > 0 ? [...current, ...additions] : current;
      });
    },
    [isJoyCodeProvider, onCatalogModelsChange],
  );

  const handleFetchModels = useCallback(() => {
    if (!codexBaseUrl || (!codexApiKey && !isJoyCodeProvider)) {
      showFetchModelsError(null, t, {
        hasApiKey: !!codexApiKey || isJoyCodeProvider,
        hasBaseUrl: !!codexBaseUrl,
      });
      return;
    }
    setIsFetchingModels(true);
    const fetcher = isJoyCodeProvider
      ? fetchJoycodeModelsForConfig(localProxyHeadersOverride)
      : fetchModelsForConfig(
          codexBaseUrl,
          codexApiKey,
          isFullUrl,
          undefined,
          customUserAgent,
        );
    fetcher
      .then((models) => {
        setFetchedModels(models);
        mergeFetchedModelsIntoCatalog(models);
        if (models.length === 0) {
          toast.info(t("providerForm.fetchModelsEmpty"));
        } else {
          toast.success(
            t("providerForm.fetchModelsSuccess", { count: models.length }),
          );
        }
      })
      .catch((err) => {
        console.warn("[ModelFetch] Failed:", err);
        showFetchModelsError(err, t);
      })
      .finally(() => setIsFetchingModels(false));
  }, [
    codexBaseUrl,
    codexApiKey,
    customUserAgent,
    isFullUrl,
    isJoyCodeProvider,
    localProxyHeadersOverride,
    mergeFetchedModelsIntoCatalog,
    t,
  ]);

  const resolveEndpointForApiFormat = useCallback(() => {
    if (apiFormat === "anthropic_messages") return "/v1/messages";
    if (apiFormat === "openai_chat") return "/v1/chat/completions";
    return "/v1/responses";
  }, [apiFormat]);

  const handleAddCatalogRow = useCallback(() => {
    if (!onCatalogModelsChange) return;
    const endpoint = resolveEndpointForApiFormat();
    setCatalogRows((current) => [
      ...current,
      createCatalogRow(
        isDevin
          ? {
              baseUrl: codexBaseUrl,
              apiKey: codexApiKey,
              endpoint,
              provider: endpoint === "/v1/messages" ? "anthropic" : "openai",
              authHeader: endpoint === "/v1/messages" ? "x-api-key" : "bearer",
              responsesMode: endpoint === "/v1/responses" ? "codex" : undefined,
            }
          : {
              endpoint,
              provider: endpoint === "/v1/messages" ? "anthropic" : "openai",
              authHeader: endpoint === "/v1/messages" ? "x-api-key" : "bearer",
              responsesMode: endpoint === "/v1/responses" ? "codex" : undefined,
            },
      ),
    ]);
  }, [
    codexApiKey,
    codexBaseUrl,
    isDevin,
    onCatalogModelsChange,
    resolveEndpointForApiFormat,
  ]);

  const handleUpdateCatalogRow = useCallback(
    (index: number, patch: Partial<CodexCatalogModel>) => {
      setCatalogRows((current) =>
        current.map((row, i) => (i === index ? { ...row, ...patch } : row)),
      );
    },
    [],
  );

  const handleRemoveCatalogRow = useCallback((index: number) => {
    setCatalogRows((current) => current.filter((_, i) => i !== index));
  }, []);

  const devinModelOptions = DEVIN_WINDSURF_MODEL_OPTIONS;
  const getDevinRoutePatch = (
    model: string,
    row?: CodexCatalogRow,
  ): Partial<CodexCatalogModel> => {
    const option = devinModelOptions.find((item) => item.model === model);
    const joycodeDefaults = isJoyCodeDevinProvider
      ? JOYCODE_DEVIN_ROUTE_DEFAULTS[model]
      : undefined;
    const isClaudeRoute = option?.family === "claude";
    const endpoint: CodexCatalogModel["endpoint"] =
      joycodeDefaults?.endpoint ??
      (isClaudeRoute
        ? "/v1/messages"
        : resolveEndpointForApiFormat());

    return {
      model,
      displayName: option?.displayName ?? model,
      upstreamModel:
        joycodeDefaults?.upstreamModel ??
        (isClaudeRoute ? "claude-sonnet-4-6" : "gpt-5.5"),
      endpoint,
      provider: endpoint === "/v1/messages" ? "anthropic" : "openai",
      authHeader:
        joycodeDefaults?.authHeader ??
        row?.authHeader ??
        (endpoint === "/v1/messages" ? "x-api-key" : DEVIN_DEFAULT_AUTH_HEADER),
      responsesMode: endpoint === "/v1/responses" ? "codex" : undefined,
      thinkingEnabled: joycodeDefaults?.thinkingEnabled ?? row?.thinkingEnabled,
      baseUrl: row?.baseUrl || codexBaseUrl,
      apiKey: row?.apiKey || codexApiKey,
    };
  };

  const renderCatalogActionButtons = (onAdd: () => void, addLabel: string) => (
    <div className="flex gap-1">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleFetchModels}
        disabled={isFetchingModels}
        className="h-7 gap-1"
      >
        {isFetchingModels ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Download className="h-3.5 w-3.5" />
        )}
        {t("providerForm.fetchModels")}
      </Button>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={onAdd}
        className="h-7 gap-1"
      >
        <Plus className="h-3.5 w-3.5" />
        {addLabel}
      </Button>
    </div>
  );

  return (
    <>
      {/* Codex API Key 输入框 */}
      <ApiKeySection
        id="codexApiKey"
        label="API Key"
        value={codexApiKey}
        onChange={onApiKeyChange}
        category={category}
        shouldShowLink={shouldShowApiKeyLink}
        websiteUrl={websiteUrl}
        isPartner={isPartner}
        partnerPromotionKey={partnerPromotionKey}
        placeholder={{
          official: t("providerForm.codexOfficialNoApiKey", {
            defaultValue: "官方供应商无需 API Key",
          }),
          thirdParty: t("providerForm.codexApiKeyAutoFill", {
            defaultValue: "输入 API Key，将自动填充到配置",
          }),
        }}
      />
      {isJoyCodeProvider && (
        <div className="flex justify-end">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleJoyCodeLogin}
            className="h-8 gap-1.5"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            {t("providerForm.loginWithJoyCode", {
              defaultValue: "登录 JoyCode",
            })}
          </Button>
        </div>
      )}

      {/* Codex Base URL 输入框 */}
      {shouldShowSpeedTest && (
        <EndpointField
          id="codexBaseUrl"
          label={t("codexConfig.apiUrlLabel")}
          value={codexBaseUrl}
          onChange={onBaseUrlChange}
          placeholder={t("providerForm.codexApiEndpointPlaceholder")}
          hint={t("providerForm.codexApiHint")}
          showFullUrlToggle
          isFullUrl={isFullUrl}
          onFullUrlChange={onFullUrlChange}
          onManageClick={() => onEndpointModalToggle(true)}
        />
      )}

      {/* 高级选项 —— 上游格式/模型映射/思考能力/自定义 UA；预设供应商通常无需展开 */}
      {category !== "official" && (
        <Collapsible
          open={advancedExpanded}
          onOpenChange={setAdvancedExpanded}
          className="rounded-lg border border-border-default p-4"
        >
          <CollapsibleTrigger asChild>
            <Button
              type="button"
              variant={null}
              size="sm"
              className="h-8 w-full justify-start gap-1.5 px-0 text-sm font-medium text-foreground hover:opacity-70"
            >
              {advancedExpanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              {t("providerForm.advancedOptionsToggle", {
                defaultValue: "高级选项",
              })}
            </Button>
          </CollapsibleTrigger>
          {!advancedExpanded && (
            <p className="mt-1 ml-1 text-xs text-muted-foreground">
              {t("codexConfig.advancedSectionHint", {
                defaultValue:
                  "包含上游格式、模型映射、思考能力与自定义 User-Agent。使用 Chat Completions 协议的供应商需开启路由接管才能使用。",
              })}
            </p>
          )}
          <CollapsibleContent className="space-y-3 pt-3">
            {/* 上游格式 —— Chat 需开启路由接管（走代理转换），Responses 原生直连。
                沿用 shouldShowSpeedTest 门控，cloud_provider 保持不可切换。 */}
            {shouldShowSpeedTest && (
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <FormLabel htmlFor="codex-upstream-format">
                    {t("codexConfig.upstreamFormatLabel", {
                      defaultValue: "上游格式",
                    })}
                  </FormLabel>
                  <Select
                    value={apiFormat}
                    onValueChange={(value) =>
                      onApiFormatChange(value as CodexApiFormat)
                    }
                  >
                    <SelectTrigger
                      id="codex-upstream-format"
                      className="w-full"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai_chat">
                        {t("codexConfig.upstreamFormatChat", {
                          defaultValue: "Chat Completions（需开启路由）",
                        })}
                      </SelectItem>
                      <SelectItem value="anthropic_messages">
                        {t("codexConfig.upstreamFormatMessages", {
                          defaultValue: "Anthropic Messages（需开启路由）",
                        })}
                      </SelectItem>
                      <SelectItem value="openai_responses">
                        {t("codexConfig.upstreamFormatResponses", {
                          defaultValue: "Responses（原生）",
                        })}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {t("codexConfig.upstreamFormatHint", {
                      defaultValue:
                        "供应商原生是 Responses API 就选 Responses；使用 OpenAI Chat 或 Anthropic Messages 协议时选择对应结构，并通过模型映射生成本地路由。",
                    })}
                  </p>
                </div>
              </div>
            )}

            {isChatFormat && canEditReasoning && (
              <div
                className={cn(
                  "space-y-3",
                  shouldShowSpeedTest && "border-t border-border-default pt-3",
                )}
              >
                <div className="space-y-1">
                  <FormLabel>
                    {t("codexConfig.reasoningGroupTitle", {
                      defaultValue: "思考能力",
                    })}
                  </FormLabel>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {t("codexConfig.reasoningSectionHint", {
                      defaultValue:
                        "预设供应商已自动配置；自定义供应商会按名称/地址自动推断。仅当自动识别不准时才需手动覆盖。",
                    })}
                  </p>
                </div>

                <div className="flex items-center justify-between gap-4">
                  <div className="space-y-1">
                    <FormLabel>
                      {t("codexConfig.reasoningModeToggle", {
                        defaultValue: "支持思考模式",
                      })}
                    </FormLabel>
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {t("codexConfig.reasoningModeHint", {
                        defaultValue:
                          "上游 Chat Completions 接口支持开启或关闭 thinking 时启用。Kimi、GLM、Qwen 等通常属于这一类。",
                      })}
                    </p>
                  </div>
                  <Switch
                    checked={supportsThinking}
                    onCheckedChange={handleReasoningThinkingChange}
                    aria-label={t("codexConfig.reasoningModeToggle", {
                      defaultValue: "支持思考模式",
                    })}
                  />
                </div>

                <div className="flex items-center justify-between gap-4 border-t border-border-default pt-3">
                  <div className="space-y-1">
                    <FormLabel>
                      {t("codexConfig.reasoningEffortToggle", {
                        defaultValue: "支持思考等级",
                      })}
                    </FormLabel>
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {t("codexConfig.reasoningEffortHint", {
                        defaultValue:
                          "上游支持 low/high/max 等思考深度控制时启用。启用后会自动启用思考模式，并把 Codex 的 reasoning.effort 转成上游 Chat 参数。",
                      })}
                    </p>
                  </div>
                  <Switch
                    checked={supportsEffort}
                    onCheckedChange={handleReasoningEffortChange}
                    aria-label={t("codexConfig.reasoningEffortToggle", {
                      defaultValue: "支持思考等级",
                    })}
                  />
                </div>
              </div>
            )}

            {/* 模型映射 / 模型目录 —— 与「路由接管」解耦，常驻显示（可编辑即渲染）。
                填了才生成 catalog：Chat 模式生成兼容路由、原生 Responses 生成
                model-catalogs.json；留空则不生成。排在自定义 UA 之前。 */}
            {canEditCatalog && (
              <div
                className={cn(
                  "space-y-4",
                  (shouldShowSpeedTest || (isChatFormat && canEditReasoning)) &&
                    "border-t border-border-default pt-3",
                )}
              >
                <div className="space-y-1">
                  <div className="flex items-center justify-between gap-3">
                    <FormLabel>
                      {t("codexConfig.modelMappingTitle", {
                        defaultValue: "模型映射",
                      })}
                    </FormLabel>
                    {renderCatalogActionButtons(
                      handleAddCatalogRow,
                      t("codexConfig.addCatalogModel", {
                        defaultValue: "添加模型",
                      }),
                    )}
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {isDevin
                      ? t("devinConfig.modelMappingHint", {
                          defaultValue:
                            "每行选择一个 Devin 请求模型，并填写对应的上游模型和端点；Base URL 和 API Key 使用上方配置。",
                        })
                      : t("codexConfig.modelMappingHint", {
                          defaultValue:
                            "选择模型角色后，CC Switch 会自动生成 Codex 兼容路由；菜单显示名可以填 DeepSeek、Kimi 等品牌模型，实际请求模型按右侧填写内容发送。",
                        })}
                  </p>
                </div>

                {catalogRows.length > 0 && (
                  <div className="space-y-2">
                    {/* 列头：md+ 显示 */}
                    <div
                      className={cn(
                        "hidden gap-2 px-1 text-xs font-medium text-muted-foreground md:grid",
                        isDevin
                          ? "grid-cols-[minmax(220px,1fr)_minmax(260px,1.4fr)_120px_36px]"
                          : "grid-cols-[1fr_1fr_140px_150px_36px]",
                      )}
                    >
                      {isDevin ? (
                        <>
                          <span>
                            {t("devinConfig.requestModel", {
                              defaultValue: "Devin 请求模型",
                            })}
                          </span>
                          <span>
                            {t("devinConfig.routeSummary", {
                              defaultValue: "上游路由",
                            })}
                          </span>
                          <span>Thinking</span>
                          <span />
                        </>
                      ) : (
                        <>
                          <span>
                            {t("codexConfig.catalogColumnDisplay", {
                              defaultValue: "菜单显示名",
                            })}
                          </span>
                          <span>
                            {t("codexConfig.catalogColumnModel", {
                              defaultValue: "实际请求模型",
                            })}
                          </span>
                          <span>
                            {t("codexConfig.catalogColumnContext", {
                              defaultValue: "上下文窗口",
                            })}
                          </span>
                          <span>
                            {t("codexConfig.catalogColumnEndpoint", {
                              defaultValue: "上游端点",
                            })}
                          </span>
                          <span />
                        </>
                      )}
                    </div>

                    {catalogRows.map((row, index) => {
                      if (isDevin) {
                        const isKnownDevinModel = devinModelOptions.some(
                          (option) => option.model === row.model,
                        );
                        return (
                          <div
                            key={row.rowId}
                            className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(220px,1fr)_minmax(260px,1.4fr)_120px_36px]"
                          >
                            <div className="grid gap-1">
                              <Select
                                value={
                                  isKnownDevinModel
                                    ? row.model
                                    : DEVIN_CUSTOM_MODEL_VALUE
                                }
                                onValueChange={(value) => {
                                  if (value === DEVIN_CUSTOM_MODEL_VALUE) {
                                    handleUpdateCatalogRow(index, {
                                      model: "",
                                      displayName: "",
                                    });
                                    return;
                                  }
                                  handleUpdateCatalogRow(
                                    index,
                                    getDevinRoutePatch(value, row),
                                  );
                                }}
                              >
                                <SelectTrigger>
                                  <SelectValue
                                    placeholder={t("devinConfig.selectModel", {
                                      defaultValue: "选择 Devin 请求模型",
                                    })}
                                  />
                                </SelectTrigger>
                                <SelectContent>
                                  {devinModelOptions.map((option) => (
                                    <SelectItem
                                      key={option.model}
                                      value={option.model}
                                    >
                                      {option.displayName}
                                    </SelectItem>
                                  ))}
                                  <SelectSeparator />
                                  <SelectItem value={DEVIN_CUSTOM_MODEL_VALUE}>
                                    {t("common.custom", {
                                      defaultValue: "自定义",
                                    })}
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                              {!isKnownDevinModel && (
                                <Input
                                  value={row.model}
                                  onChange={(event) =>
                                    handleUpdateCatalogRow(index, {
                                      model: event.target.value,
                                      displayName: event.target.value,
                                    })
                                  }
                                  placeholder="MODEL_CLAUDE_4_SONNET_BYOK"
                                  aria-label={t("devinConfig.requestModel", {
                                    defaultValue: "Devin 请求模型",
                                  })}
                                />
                              )}
                            </div>
                            <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(120px,1fr)_170px]">
                              <div className="flex gap-1">
                                <Input
                                  value={row.upstreamModel ?? ""}
                                  onChange={(event) =>
                                    handleUpdateCatalogRow(index, {
                                      upstreamModel: event.target.value,
                                    })
                                  }
                                  placeholder="gpt-5.5"
                                  aria-label={t("devinConfig.upstreamModel", {
                                    defaultValue: "上游模型",
                                  })}
                                  className="flex-1"
                                />
                                {fetchedModels.length > 0 && (
                                  <ModelDropdown
                                    models={fetchedModels}
                                    onSelect={(id) =>
                                      handleUpdateCatalogRow(index, {
                                        upstreamModel: id,
                                      })
                                    }
                                  />
                                )}
                              </div>
                              <Select
                                value={row.endpoint ?? "/v1/responses"}
                                onValueChange={(value) =>
                                  handleUpdateCatalogRow(index, {
                                    endpoint:
                                      value as CodexCatalogModel["endpoint"],
                                    provider:
                                      value === "/v1/messages"
                                        ? "anthropic"
                                        : "openai",
                                    authHeader:
                                      row.authHeader ??
                                      DEVIN_DEFAULT_AUTH_HEADER,
                                  })
                                }
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="/v1/responses">
                                    /v1/responses
                                  </SelectItem>
                                  <SelectItem value="/v1/chat/completions">
                                    /v1/chat/completions
                                  </SelectItem>
                                  <SelectItem value="/v1/messages">
                                    /v1/messages
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="flex h-9 items-center justify-between gap-2 rounded-md border border-border-default px-3">
                              <span className="text-xs text-muted-foreground">
                                Thinking
                              </span>
                              <Switch
                                checked={row.thinkingEnabled !== false}
                                onCheckedChange={(checked) =>
                                  handleUpdateCatalogRow(index, {
                                    thinkingEnabled: checked
                                      ? undefined
                                      : false,
                                  })
                                }
                                aria-label="Thinking"
                              />
                            </div>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="h-9 w-9 text-muted-foreground hover:text-destructive"
                              onClick={() => handleRemoveCatalogRow(index)}
                              title={t("common.delete", {
                                defaultValue: "删除",
                              })}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        );
                      }

                      return (
                        <div
                          key={row.rowId}
                          className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_1fr_140px_150px_36px]"
                        >
                          <Input
                            value={row.displayName ?? ""}
                            onChange={(event) =>
                              handleUpdateCatalogRow(index, {
                                displayName: event.target.value,
                              })
                            }
                            placeholder={t(
                              "codexConfig.catalogDisplayNamePlaceholder",
                              {
                                defaultValue: "例如: DeepSeek V4 Flash",
                              },
                            )}
                            aria-label={t("codexConfig.catalogColumnDisplay", {
                              defaultValue: "菜单显示名",
                            })}
                          />
                          <div className="flex gap-1">
                            <Input
                              value={row.model}
                              onChange={(event) =>
                                handleUpdateCatalogRow(index, {
                                  model: event.target.value,
                                })
                              }
                              placeholder={t(
                                "codexConfig.catalogModelPlaceholder",
                                {
                                  defaultValue: "例如: deepseek-v4-flash",
                                },
                              )}
                              aria-label={t("codexConfig.catalogColumnModel", {
                                defaultValue: "实际请求模型",
                              })}
                              className="flex-1"
                            />
                            {fetchedModels.length > 0 && (
                              <ModelDropdown
                                models={fetchedModels}
                                onSelect={(id) =>
                                  handleUpdateCatalogRow(index, {
                                    model: id,
                                    displayName: row.displayName?.trim()
                                      ? row.displayName
                                      : id,
                                  })
                                }
                              />
                            )}
                          </div>
                          <Input
                            type="number"
                            min={1}
                            inputMode="numeric"
                            value={row.contextWindow ?? ""}
                            onChange={(event) =>
                              handleUpdateCatalogRow(index, {
                                contextWindow: event.target.value.replace(
                                  /[^\d]/g,
                                  "",
                                ),
                              })
                            }
                            placeholder={t(
                              "codexConfig.contextWindowPlaceholder",
                              {
                                defaultValue: "例如: 128000",
                              },
                            )}
                            aria-label={t("codexConfig.catalogColumnContext", {
                              defaultValue: "上下文窗口",
                            })}
                          />
                          <Select
                            value={row.endpoint ?? "default"}
                            onValueChange={(value) => {
                              if (value === "default") {
                                handleUpdateCatalogRow(index, {
                                  endpoint: undefined,
                                  provider: undefined,
                                  authHeader: undefined,
                                });
                                return;
                              }
                              const endpoint =
                                value as CodexCatalogModel["endpoint"];
                              handleUpdateCatalogRow(index, {
                                endpoint,
                                provider:
                                  endpoint === "/v1/messages"
                                    ? "anthropic"
                                    : "openai",
                                authHeader:
                                  endpoint === "/v1/messages"
                                    ? "x-api-key"
                                    : "bearer",
                              });
                            }}
                          >
                            <SelectTrigger
                              aria-label={t(
                                "codexConfig.catalogColumnEndpoint",
                                {
                                  defaultValue: "上游端点",
                                },
                              )}
                            >
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="default">
                                {t("common.default", {
                                  defaultValue: "默认",
                                })}
                              </SelectItem>
                              <SelectItem value="/v1/responses">
                                /v1/responses
                              </SelectItem>
                              <SelectItem value="/v1/chat/completions">
                                /v1/chat/completions
                              </SelectItem>
                              <SelectItem value="/v1/messages">
                                /v1/messages
                              </SelectItem>
                            </SelectContent>
                          </Select>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-9 w-9 text-muted-foreground hover:text-destructive"
                            onClick={() => handleRemoveCatalogRow(index)}
                            title={t("common.delete", {
                              defaultValue: "删除",
                            })}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            <div
              className={cn(
                "space-y-3",
                (shouldShowSpeedTest ||
                  (isChatFormat && canEditReasoning) ||
                  canEditCatalog) &&
                  "border-t border-border-default pt-3",
              )}
            >
              <CustomUserAgentField
                id="codex-custom-user-agent"
                value={customUserAgent}
                onChange={onCustomUserAgentChange}
              />
              <div className="border-t border-border-default pt-3">
                <LocalProxyRequestOverridesField
                  headersJson={localProxyHeadersOverride}
                  bodyJson={localProxyBodyOverride}
                  onHeadersJsonChange={onLocalProxyHeadersOverrideChange}
                  onBodyJsonChange={onLocalProxyBodyOverrideChange}
                />
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* 端点测速弹窗 - Codex */}
      {shouldShowSpeedTest && isEndpointModalOpen && (
        <EndpointSpeedTest
          appId="codex"
          providerId={providerId}
          value={codexBaseUrl}
          onChange={onBaseUrlChange}
          initialEndpoints={speedTestEndpoints}
          visible={isEndpointModalOpen}
          onClose={() => onEndpointModalToggle(false)}
          autoSelect={autoSelect}
          onAutoSelectChange={onAutoSelectChange}
          onCustomEndpointsChange={onCustomEndpointsChange}
        />
      )}
    </>
  );
}

import {
  codexProviderPresets,
  type CodexProviderPreset,
  generateThirdPartyAuth,
  generateThirdPartyConfig,
} from "./codexProviderPresets";
import type { CodexCatalogModel } from "@/types";

export type DevinProviderPreset = CodexProviderPreset;

const ANYROUTER_KEY = "sk-8qR928CeKLhl1ZLKKk7DpTCSlzzYqJPTJO5ZrFtnIpdjhQoQ";
const MUYUAN_KEY = "sk-oGdoeLsjwQ19tlszs65Om61dAwSdfstjeOmFFHSR4g5KKMHx";
const JOYCODE_KEY = "sk-joycode-proxy";
const LLMHUB_QZZ_KEY =
  "sk-4a83c1cf22efc3fd647b3d7b9c8640ccfcd6dbf26cf60ce1b507ca289d9b4b04";
const ANYROUTER_CLAUDE_MODEL = "claude-haiku-4-5-20251001";
const DEVIN_THINKING_REASONING: CodexProviderPreset["codexChatReasoning"] = {
  supportsThinking: true,
  supportsEffort: true,
  thinkingParam: "thinking",
  effortParam: "reasoning_effort",
  effortValueMode: "passthrough",
  outputFormat: "reasoning_content",
};

export const DEVIN_WINDSURF_MODEL_OPTIONS = [
  // Keep Devin's menu small: only the routes confirmed usable by the local bridge.
  {
    model: "swe-1-6-slow",
    displayName: "SWE 1.6 Slow",
    family: "openai",
  },
  {
    model: "MODEL_CLAUDE_4_SONNET_BYOK",
    displayName: "Claude Sonnet 4 BYOK",
    family: "claude",
  },
  {
    model: "MODEL_CLAUDE_4_OPUS_THINKING_BYOK",
    displayName: "Claude Opus 4 Thinking BYOK",
    family: "claude",
  },
  {
    model: "MODEL_CLAUDE_4_OPUS_BYOK",
    displayName: "Claude Opus 4 BYOK",
    family: "claude",
  },
] as const;

export interface ModelGroup {
  label: string;
  models: Array<{
    model: string;
    displayName: string;
    family: string;
  }>;
}

/**
 * 将模型选项按系列分组，用于 UI 展示
 */
export function getGroupedModelOptions(): ModelGroup[] {
  return [
    {
      label: "Available",
      models: DEVIN_WINDSURF_MODEL_OPTIONS.map((model) => ({ ...model })),
    },
  ];
}

const windsurfOpenAiAliases = DEVIN_WINDSURF_MODEL_OPTIONS.filter(
  (option) => option.family === "openai",
).map((option) => option.model);

const windsurfClaudeAliases = DEVIN_WINDSURF_MODEL_OPTIONS.filter(
  (option) => option.family === "claude",
).map((option) => option.model);

const windsurfAllAliases = [
  ...windsurfOpenAiAliases,
  ...windsurfClaudeAliases,
] as const;

function devinModelRoute({
  model,
  displayName,
  contextWindow,
  upstreamModel,
  endpoint,
  baseUrl,
  apiKey,
  authHeader = endpoint === "/v1/messages" ? "x-api-key" : "bearer",
  headers,
  responsesMode,
  thinkingEnabled,
}: {
  model: string;
  displayName: string;
  contextWindow?: string | number;
  upstreamModel?: string;
  endpoint: NonNullable<CodexCatalogModel["endpoint"]>;
  baseUrl: string;
  apiKey: string;
  authHeader?: NonNullable<CodexCatalogModel["authHeader"]>;
  headers?: Record<string, string>;
  responsesMode?: CodexCatalogModel["responsesMode"];
  thinkingEnabled?: boolean;
}): CodexCatalogModel {
  const isMessagesEndpoint = endpoint.endsWith("/messages");
  const builtinModel = DEVIN_WINDSURF_MODEL_OPTIONS.find(
    (option) => option.model === model,
  );
  const responsesFields =
    endpoint.endsWith("/responses") && responsesMode ? { responsesMode } : {};
  const thinkingFields =
    thinkingEnabled !== undefined ? { thinkingEnabled } : {};

  return {
    model,
    displayName: builtinModel?.displayName ?? displayName,
    contextWindow,
    upstreamModel: upstreamModel ?? model,
    provider: isMessagesEndpoint ? "anthropic" : "openai",
    endpoint,
    baseUrl,
    apiKey,
    authHeader,
    ...(headers && Object.keys(headers).length > 0 ? { headers } : {}),
    ...responsesFields,
    ...thinkingFields,
    routes: [
      {
        name: "primary",
        baseUrl,
        apiKey,
        enabled: true,
        priority: 10,
        authHeader,
        ...(headers && Object.keys(headers).length > 0 ? { headers } : {}),
        ...responsesFields,
        ...thinkingFields,
      },
    ],
  };
}

function windsurfOpenAiCompatibleRoutes({
  baseUrl,
  apiKey,
  endpoint,
  upstreamModel,
  authHeader = "bearer",
  headers,
  responsesMode,
}: {
  baseUrl: string;
  apiKey: string;
  endpoint: NonNullable<CodexCatalogModel["endpoint"]>;
  upstreamModel: string;
  authHeader?: NonNullable<CodexCatalogModel["authHeader"]>;
  headers?: Record<string, string>;
  responsesMode?: CodexCatalogModel["responsesMode"];
}): CodexCatalogModel[] {
  return windsurfAllAliases.map((model) =>
    devinModelRoute({
      model,
      displayName: `Windsurf ${model}`,
      upstreamModel,
      endpoint,
      baseUrl,
      apiKey,
      authHeader,
      headers,
      responsesMode,
    }),
  );
}

function windsurfModelRoutes({
  baseUrl,
  apiKey,
  openAiEndpoint,
  openAiUpstreamModel,
  openAiAuthHeader = "bearer",
  openAiResponsesMode,
  openAiHeaders,
  claudeEndpoint = "/v1/messages",
  claudeUpstreamModel,
  claudeAliasUpstreamModels,
  claudeAuthHeader,
  claudeHeaders,
}: {
  baseUrl: string;
  apiKey: string;
  openAiEndpoint: NonNullable<CodexCatalogModel["endpoint"]>;
  openAiUpstreamModel: string;
  openAiAuthHeader?: NonNullable<CodexCatalogModel["authHeader"]>;
  openAiResponsesMode?: CodexCatalogModel["responsesMode"];
  openAiHeaders?: Record<string, string>;
  claudeEndpoint?: NonNullable<CodexCatalogModel["endpoint"]>;
  claudeUpstreamModel: string;
  claudeAliasUpstreamModels?: Partial<Record<string, string>>;
  claudeAuthHeader: NonNullable<CodexCatalogModel["authHeader"]>;
  claudeHeaders?: Record<string, string>;
}): CodexCatalogModel[] {
  return [
    ...windsurfOpenAiAliases.map((model) =>
      devinModelRoute({
        model,
        displayName: `Windsurf ${model}`,
        upstreamModel: openAiUpstreamModel,
        endpoint: openAiEndpoint,
        baseUrl,
        apiKey,
        authHeader: openAiAuthHeader,
        headers: openAiHeaders,
        responsesMode: openAiResponsesMode,
      }),
    ),
    ...windsurfClaudeAliases.map((model) =>
      devinModelRoute({
        model,
        displayName: `Windsurf ${model}`,
        upstreamModel:
          claudeAliasUpstreamModels?.[model] ?? claudeUpstreamModel,
        endpoint: claudeEndpoint,
        baseUrl,
        apiKey,
        authHeader: claudeAuthHeader,
        headers: claudeHeaders,
      }),
    ),
  ];
}

const localTapModels: CodexCatalogModel[] = [
  devinModelRoute({
    model: "swe-1-6-slow",
    displayName: "SWE 1.6 Slow",
    upstreamModel: "GPT 5.3-codex",
    endpoint: "/api/saas/openai/v1/responses",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
    responsesMode: "codex",
  }),
  devinModelRoute({
    model: "Claude Sonnet 4 Thinking BYOK",
    displayName: "Claude Sonnet 4 Thinking BYOK",
    upstreamModel: "GLM-5.1",
    endpoint: "/v1/chat/completions",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
  }),
  devinModelRoute({
    model: "MODEL_CLAUDE_4_SONNET_THINKING_BYOK",
    displayName: "Claude Sonnet 4 Thinking BYOK",
    upstreamModel: "GLM-5.1",
    endpoint: "/v1/chat/completions",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
  }),
  devinModelRoute({
    model: "MODEL_CLAUDE_4_SONNET_BYOK",
    displayName: "Claude Sonnet 4 BYOK",
    upstreamModel: "GLM-5.1",
    endpoint: "/v1/chat/completions",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
  }),
  devinModelRoute({
    model: "MODEL_CLAUDE_4_OPUS_THINKING_BYOK",
    displayName: "Claude Opus 4 Thinking BYOK",
    upstreamModel: "GLM-5.1",
    endpoint: "/v1/chat/completions",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
  }),
  devinModelRoute({
    model: "MODEL_CLAUDE_4_OPUS_BYOK",
    displayName: "Claude Opus 4 BYOK",
    upstreamModel: "GLM-5.1",
    endpoint: "/v1/chat/completions",
    baseUrl: "https://joycode-api.jd.com",
    apiKey: JOYCODE_KEY,
    authHeader: "bearer",
  }),
];

function parseTomlStringField(config: string, key: string): string | undefined {
  const match = config.match(
    new RegExp(`(^|\\n)\\s*${key}\\s*=\\s*"((?:\\\\.|[^"\\\\])*)"`),
  );
  if (!match) return undefined;
  try {
    return JSON.parse(`"${match[2]}"`);
  } catch {
    return match[2];
  }
}

function resolvePresetBaseUrl(preset: CodexProviderPreset): string {
  return (
    preset.endpointCandidates?.[0] ??
    parseTomlStringField(preset.config, "base_url") ??
    "https://api.openai.com/v1"
  ).trimEnd();
}

function resolvePresetModel(preset: CodexProviderPreset): string {
  return (
    preset.modelCatalog?.[0]?.model ??
    parseTomlStringField(preset.config, "model") ??
    "gpt-5.5"
  );
}

function resolvePresetApiKey(preset: CodexProviderPreset): string {
  const value = preset.auth?.OPENAI_API_KEY;
  return typeof value === "string" ? value : "";
}

function resolvePresetEndpoint(
  preset: CodexProviderPreset,
): NonNullable<CodexCatalogModel["endpoint"]> {
  return preset.apiFormat === "openai_chat"
    ? "/v1/chat/completions"
    : "/v1/responses";
}

function devinCatalogFromCodexPreset(
  preset: CodexProviderPreset,
  options: {
    baseUrl?: string;
    apiKey?: string;
    endpoint?: NonNullable<CodexCatalogModel["endpoint"]>;
  } = {},
): CodexCatalogModel[] {
  const baseUrl = (options.baseUrl ?? resolvePresetBaseUrl(preset)).trimEnd();
  const apiKey = options.apiKey ?? resolvePresetApiKey(preset);
  const endpoint = options.endpoint ?? resolvePresetEndpoint(preset);
  const responsesMode = endpoint === "/v1/responses" ? "codex" : undefined;
  const catalogModels =
    preset.modelCatalog && preset.modelCatalog.length > 0
      ? preset.modelCatalog
      : [
          {
            model: resolvePresetModel(preset),
            displayName: resolvePresetModel(preset),
          },
        ];

  const defaultUpstreamModel =
    catalogModels[0]?.upstreamModel ?? catalogModels[0]?.model ?? "gpt-5.5";

  return windsurfOpenAiCompatibleRoutes({
    baseUrl,
    apiKey,
    endpoint,
    upstreamModel: defaultUpstreamModel,
    responsesMode,
  });
}

function devinPresetFromCodexPreset(
  preset: CodexProviderPreset,
): DevinProviderPreset | undefined {
  if (preset.isOfficial || preset.category === "official") return undefined;

  const baseUrl = resolvePresetBaseUrl(preset);
  const model = resolvePresetModel(preset);
  return {
    ...preset,
    config: generateThirdPartyConfig(preset.name, baseUrl, model),
    apiFormat: preset.apiFormat ?? "openai_responses",
    codexChatReasoning: preset.codexChatReasoning ?? DEVIN_THINKING_REASONING,
    modelCatalog: devinCatalogFromCodexPreset(preset, { baseUrl }),
  };
}

const dedicatedDevinPresetNames = new Set([
  "JoyCode",
  "Any Router",
  "Muyuan",
  "LLMHub QZZ",
]);

const codexBackfillDevinPresets = codexProviderPresets
  .filter((preset) => !dedicatedDevinPresetNames.has(preset.name))
  .map(devinPresetFromCodexPreset)
  .filter((preset): preset is DevinProviderPreset => Boolean(preset));

export const devinProviderPresets: DevinProviderPreset[] = [
  {
    name: "JoyCode",
    websiteUrl: "https://joycode-api.jd.com",
    auth: generateThirdPartyAuth("sk-joycode-proxy"),
    config: generateThirdPartyConfig(
      "joycode",
      "https://joycode-api.jd.com",
      "MODEL_CLAUDE_4_SONNET_BYOK",
    ),
    category: "custom",
    endpointCandidates: ["https://joycode-api.jd.com"],
    icon: "openai",
    apiFormat: "openai_responses",
    codexChatReasoning: DEVIN_THINKING_REASONING,
    modelCatalog: localTapModels,
  },
  {
    name: "Any Router",
    websiteUrl: "https://anyrouter.top",
    auth: generateThirdPartyAuth(ANYROUTER_KEY),
    config: generateThirdPartyConfig(
      "Any Router",
      "https://a-ocnfniawgw.cn-shanghai.fcapp.run/v1",
      "MODEL_CLAUDE_4_SONNET_BYOK",
    ),
    category: "aggregator",
    endpointCandidates: [
      "https://a-ocnfniawgw.cn-shanghai.fcapp.run/v1",
      "https://anyrouter.top/v1",
    ],
    icon: "openai",
    apiFormat: "openai_responses",
    codexChatReasoning: DEVIN_THINKING_REASONING,
    modelCatalog: [
      ...windsurfModelRoutes({
        baseUrl: "https://a-ocnfniawgw.cn-shanghai.fcapp.run/v1",
        apiKey: ANYROUTER_KEY,
        openAiEndpoint: "/v1/chat/completions",
        openAiUpstreamModel: "gpt-5.5",
        claudeEndpoint: "/v1/chat/completions",
        claudeUpstreamModel: ANYROUTER_CLAUDE_MODEL,
        claudeAliasUpstreamModels: {
          MODEL_CLAUDE_4_OPUS_BYOK: ANYROUTER_CLAUDE_MODEL,
          MODEL_CLAUDE_4_OPUS_THINKING_BYOK: ANYROUTER_CLAUDE_MODEL,
        },
        claudeAuthHeader: "bearer",
      }),
    ],
  },
  {
    name: "Muyuan",
    websiteUrl: "https://muyuan.do",
    auth: generateThirdPartyAuth(MUYUAN_KEY),
    config: generateThirdPartyConfig(
      "muyuan",
      "https://muyuan.do/v1",
      "MODEL_CLAUDE_4_SONNET_BYOK",
    ),
    category: "aggregator",
    endpointCandidates: ["https://muyuan.do/v1"],
    icon: "newapi",
    apiFormat: "openai_chat",
    codexChatReasoning: DEVIN_THINKING_REASONING,
    modelCatalog: [
      ...windsurfModelRoutes({
        baseUrl: "https://muyuan.do",
        apiKey: MUYUAN_KEY,
        openAiEndpoint: "/v1/chat/completions",
        openAiUpstreamModel: "gpt-5.5",
        claudeEndpoint: "/v1/chat/completions",
        claudeUpstreamModel: "claude-sonnet-4-6",
        claudeAliasUpstreamModels: {
          MODEL_CLAUDE_4_OPUS_BYOK: "claude-sonnet-4-6",
          MODEL_CLAUDE_4_OPUS_THINKING_BYOK: "claude-sonnet-4-6",
        },
        claudeAuthHeader: "bearer",
      }),
    ],
  },
  {
    name: "LLMHub QZZ",
    websiteUrl: "https://llmhub.qzz.io",
    auth: generateThirdPartyAuth(LLMHUB_QZZ_KEY),
    config: generateThirdPartyConfig(
      "llmhub_qzz",
      "https://llmhub.qzz.io/v1",
      "MODEL_CLAUDE_4_SONNET_BYOK",
    ),
    category: "aggregator",
    endpointCandidates: ["https://llmhub.qzz.io/v1"],
    icon: "newapi",
    apiFormat: "openai_chat",
    codexChatReasoning: DEVIN_THINKING_REASONING,
    modelCatalog: [
      ...windsurfModelRoutes({
        baseUrl: "https://llmhub.qzz.io",
        apiKey: LLMHUB_QZZ_KEY,
        openAiEndpoint: "/v1/chat/completions",
        openAiUpstreamModel: "gpt-5.5",
        claudeEndpoint: "/v1/chat/completions",
        claudeUpstreamModel: "claude-sonnet-4-6",
        claudeAliasUpstreamModels: {
          MODEL_CLAUDE_4_OPUS_BYOK: "claude-sonnet-4-6",
          MODEL_CLAUDE_4_OPUS_THINKING_BYOK: "claude-sonnet-4-6",
        },
        claudeAuthHeader: "bearer",
      }),
    ],
  },
  ...codexBackfillDevinPresets,
];

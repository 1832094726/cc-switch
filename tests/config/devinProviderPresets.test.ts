import { describe, expect, it } from "vitest";
import { codexProviderPresets } from "@/config/codexProviderPresets";
import {
  DEVIN_WINDSURF_MODEL_OPTIONS,
  devinProviderPresets,
} from "@/config/devinProviderPresets";

describe("devinProviderPresets", () => {
  it("keeps dedicated Devin presets and backfills the Codex provider catalog", () => {
    const dedicatedPresetNames = new Set([
      "JoyCode",
      "Any Router",
      "Muyuan",
      "LLMHub QZZ",
    ]);
    const codexBackfillCount = codexProviderPresets.filter(
      (preset) =>
        !preset.isOfficial &&
        preset.category !== "official" &&
        !dedicatedPresetNames.has(preset.name),
    ).length;

    expect(devinProviderPresets.length).toBeGreaterThanOrEqual(
      codexBackfillCount + 4,
    );
    expect(devinProviderPresets.map((preset) => preset.name)).toEqual(
      expect.arrayContaining([
        "JoyCode",
        "Any Router",
        "Muyuan",
        "LLMHub QZZ",
        "DeepSeek",
        "SiliconFlow",
        "OpenRouter",
      ]),
    );
    for (const name of [
      "JoyCode",
      "Any Router",
      "Muyuan",
      "LLMHub QZZ",
    ]) {
      expect(
        devinProviderPresets.filter((preset) => preset.name === name),
        name,
      ).toHaveLength(1);
    }
  });

  it("gives every Devin preset a routed Windsurf model catalog", () => {
    for (const preset of devinProviderPresets) {
      expect(preset.modelCatalog?.length, preset.name).toBeGreaterThan(0);
      expect(
        preset.modelCatalog?.some(
          (model) =>
            model.model.startsWith("MODEL_") || model.model.startsWith("swe-"),
        ),
        preset.name,
      ).toBe(true);
      expect(
        preset.modelCatalog?.every((model) =>
          [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/messages",
            "/api/saas/openai/v1/responses",
            "/api/saas/anthropic/v1/messages",
          ].includes(model.endpoint ?? ""),
        ),
        preset.name,
      ).toBe(true);
    }

    expect(DEVIN_WINDSURF_MODEL_OPTIONS).toEqual([
      expect.objectContaining({
        model: "swe-1-6-slow",
        displayName: "SWE 1.6 Slow",
      }),
      expect.objectContaining({
        model: "MODEL_CLAUDE_4_SONNET_BYOK",
        displayName: "Claude Sonnet 4 BYOK",
      }),
      expect.objectContaining({
        model: "MODEL_CLAUDE_4_OPUS_THINKING_BYOK",
        displayName: "Claude Opus 4 Thinking BYOK",
      }),
      expect.objectContaining({
        model: "MODEL_CLAUDE_4_OPUS_BYOK",
        displayName: "Claude Opus 4 BYOK",
      }),
    ]);
  });

  it("routes JoyCode Devin models to JoyCode-owned model ids", () => {
    const joycode = devinProviderPresets.find(
      (preset) => preset.name === "JoyCode",
    );

    expect(
      new Set(joycode?.modelCatalog?.map((model) => model.endpoint)),
    ).toEqual(
      new Set([
        "/api/saas/openai/v1/responses",
        "/api/saas/anthropic/v1/messages",
      ]),
    );
    expect(
      joycode?.modelCatalog?.find((model) => model.model === "swe-1-6-slow"),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/api/saas/openai/v1/responses",
        upstreamModel: "GPT 5.3-codex",
        authHeader: "bearer",
      }),
    );
    expect(
      joycode?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_SONNET_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/api/saas/anthropic/v1/messages",
        upstreamModel: "Claude-Sonnet-4.6-hq",
        authHeader: "x-api-key",
      }),
    );
    expect(
      joycode?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_SONNET_BYOK",
      )?.thinkingEnabled,
    ).toBeUndefined();
    expect(
      joycode?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_OPUS_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/api/saas/anthropic/v1/messages",
        upstreamModel: "Claude-Opus-4.6-hq",
        authHeader: "x-api-key",
      }),
    );
  });

  it("preserves multi-endpoint routing for providers that expose Claude and OpenAI separately", () => {
    const anyRouter = devinProviderPresets.find(
      (preset) => preset.name === "Any Router",
    );
    const muyuan = devinProviderPresets.find(
      (preset) => preset.name === "Muyuan",
    );
    const llmhub = devinProviderPresets.find(
      (preset) => preset.name === "LLMHub QZZ",
    );

    expect(
      new Set(anyRouter?.modelCatalog?.map((model) => model.endpoint)),
    ).toEqual(new Set(["/v1/chat/completions"]));
    expect(
      new Set(muyuan?.modelCatalog?.map((model) => model.endpoint)),
    ).toEqual(new Set(["/v1/chat/completions"]));
    expect(
      new Set(llmhub?.modelCatalog?.map((model) => model.endpoint)),
    ).toEqual(new Set(["/v1/chat/completions"]));
    expect(llmhub?.endpointCandidates).toEqual(["https://llmhub.qzz.io/v1"]);
    expect(
      anyRouter?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_SONNET_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-haiku-4-5-20251001",
      }),
    );
    expect(
      anyRouter?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_OPUS_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-haiku-4-5-20251001",
      }),
    );
    expect(
      muyuan?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_SONNET_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-sonnet-4-6",
      }),
    );
    expect(
      muyuan?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_OPUS_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-sonnet-4-6",
      }),
    );
    expect(
      muyuan?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_OPUS_THINKING_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-sonnet-4-6",
      }),
    );
    expect(
      llmhub?.modelCatalog?.find(
        (model) => model.model === "MODEL_CLAUDE_4_OPUS_BYOK",
      ),
    ).toEqual(
      expect.objectContaining({
        endpoint: "/v1/chat/completions",
        upstreamModel: "claude-sonnet-4-6",
      }),
    );
    expect(anyRouter?.codexChatReasoning).toEqual(
      expect.objectContaining({
        supportsThinking: true,
        supportsEffort: true,
      }),
    );
  });

  it("adds Muyuan to the Codex provider presets", () => {
    const muyuan = codexProviderPresets.find(
      (preset) => preset.name === "Muyuan",
    );

    expect(muyuan).toEqual(
      expect.objectContaining({
        websiteUrl: "https://muyuan.do",
        endpointCandidates: ["https://muyuan.do/v1"],
        apiFormat: "openai_chat",
      }),
    );
  });
});

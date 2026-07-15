import { describe, expect, it } from "vitest";
import {
  normalizeCodexCatalogModelsForSave,
  shouldApplyLocalProxyRequestOverridesForApp,
} from "@/components/providers/forms/ProviderForm";

describe("ProviderForm Codex catalog helpers", () => {
  it("persists local proxy overrides for non-official Devin providers", () => {
    expect(shouldApplyLocalProxyRequestOverridesForApp("devin", "custom")).toBe(
      true,
    );
    expect(
      shouldApplyLocalProxyRequestOverridesForApp("devin", "official"),
    ).toBe(false);
  });

  it("normalizes catalog rows and removes empty or duplicate models", () => {
    expect(
      normalizeCodexCatalogModelsForSave([
        { model: " deepseek-v4-flash ", displayName: " DeepSeek " },
        { model: "deepseek-v4-flash", displayName: "Duplicate" },
        { model: "", displayName: "Empty" },
        { model: "kimi-k2", contextWindow: "128000 tokens" },
      ]),
    ).toEqual([
      { model: "deepseek-v4-flash", displayName: "DeepSeek" },
      { model: "kimi-k2", contextWindow: 128000 },
    ]);
  });

  it("preserves Devin Responses routing compatibility fields", () => {
    expect(
      normalizeCodexCatalogModelsForSave([
        {
          model: " MODEL_PRIVATE_11 ",
          upstreamModel: " gpt-5.5 ",
          endpoint: "/v1/responses",
          baseUrl: " https://example.com ",
          apiKey: " sk-test ",
          authHeader: "bearer",
          responsesMode: "codex",
        },
      ]),
    ).toEqual([
      {
        model: "MODEL_PRIVATE_11",
        upstreamModel: "gpt-5.5",
        provider: "openai",
        endpoint: "/v1/responses",
        baseUrl: "https://example.com",
        apiKey: "sk-test",
        authHeader: "bearer",
        responsesMode: "codex",
        routes: [
          {
            name: "primary",
            baseUrl: "https://example.com",
            apiKey: "sk-test",
            enabled: true,
            priority: 10,
            authHeader: "bearer",
            responsesMode: "codex",
          },
        ],
      },
    ]);
  });

  it("preserves Codex Anthropic Messages routing fields", () => {
    expect(
      normalizeCodexCatalogModelsForSave([
        {
          model: " claude-sonnet-4-6 ",
          displayName: " Claude Sonnet ",
          endpoint: "/v1/messages",
          provider: "anthropic",
          authHeader: "x-api-key",
        },
      ]),
    ).toEqual([
      {
        model: "claude-sonnet-4-6",
        displayName: "Claude Sonnet",
        provider: "anthropic",
        endpoint: "/v1/messages",
        authHeader: "x-api-key",
      },
    ]);
  });

  it("preserves native-profile overrides (parallel tool calls + input modalities + base instructions)", () => {
    expect(
      normalizeCodexCatalogModelsForSave([
        {
          model: "MiniMax-M3",
          displayName: "MiniMax-M3",
          contextWindow: 1000000,
          supportsParallelToolCalls: true,
          inputModalities: ["text", "image"],
          baseInstructions:
            "  You are Codex, a coding agent based on MiniMax-M3.  ",
        },
        // false must be preserved (not dropped as falsy); empty modalities dropped;
        // empty/whitespace baseInstructions dropped
        {
          model: "mimo-v2.5-pro",
          supportsParallelToolCalls: false,
          inputModalities: [],
          baseInstructions: "   ",
        },
      ]),
    ).toEqual([
      {
        model: "MiniMax-M3",
        displayName: "MiniMax-M3",
        contextWindow: 1000000,
        supportsParallelToolCalls: true,
        inputModalities: ["text", "image"],
        baseInstructions: "You are Codex, a coding agent based on MiniMax-M3.",
      },
      { model: "mimo-v2.5-pro", supportsParallelToolCalls: false },
    ]);
  });
});

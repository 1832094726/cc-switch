import { describe, expect, it } from "vitest";
import { normalizeCodexCatalogModelsForSave } from "@/components/providers/forms/ProviderForm";

describe("ProviderForm Codex catalog helpers", () => {
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
});

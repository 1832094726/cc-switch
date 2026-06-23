import { describe, expect, it } from "vitest";
import {
  isCodexRemoteCompactionEnabled,
  setCodexRemoteCompaction,
  updateCodexExperimentalBearerToken,
} from "./providerConfigUtils";

describe("Codex remote compaction config helpers", () => {
  it("enables remote compaction by naming the active custom provider OpenAI", () => {
    const input = `model_provider = "custom"
model = "gpt-5.4"

[model_providers.custom]
name = "AIHubMix"
base_url = "https://aihubmix.example/v1"
wire_api = "responses"

[model_providers.backup]
name = "Backup"
base_url = "https://backup.example/v1"
`;

    const result = setCodexRemoteCompaction(input, true, "AIHubMix");

    expect(isCodexRemoteCompactionEnabled(result)).toBe(true);
    expect(result).toContain(`[model_providers.custom]\nname = "OpenAI"`);
    expect(result).toContain(`[model_providers.backup]\nname = "Backup"`);
  });

  it("disables remote compaction by restoring the provider display name", () => {
    const input = `model_provider = "custom"

[model_providers.custom]
name = "OpenAI"
base_url = "https://aihubmix.example/v1"
wire_api = "responses"
`;

    const result = setCodexRemoteCompaction(input, false, "AIHubMix");

    expect(isCodexRemoteCompactionEnabled(result)).toBe(false);
    expect(result).toContain(`name = "AIHubMix"`);
  });

  it("does not rewrite reserved built-in providers", () => {
    const input = `model_provider = "openai"
model = "gpt-5"
`;

    expect(setCodexRemoteCompaction(input, true, "OpenAI")).toBe(input);
    expect(isCodexRemoteCompactionEnabled(input)).toBe(false);
  });
});

describe("Codex experimental bearer token helpers", () => {
  it("updates the active provider token when saving a new API key", () => {
    const input = `model_provider = "pipi"
model = "gpt-5.5"

[model_providers.pipi]
name = "pipi"
base_url = "https://cn.picpi.top/v1"
wire_api = "responses"
experimental_bearer_token = "sk-old"

[model_providers.backup]
name = "backup"
experimental_bearer_token = "sk-backup"
`;

    const result = updateCodexExperimentalBearerToken(input, "sk-new");

    expect(result).toContain(`experimental_bearer_token = "sk-new"`);
    expect(result).toContain(`experimental_bearer_token = "sk-backup"`);
    expect(result).not.toContain("sk-old");
  });

  it("removes the active provider token when the API key is cleared", () => {
    const input = `model_provider = "pipi"

[model_providers.pipi]
name = "pipi"
experimental_bearer_token = "sk-old"
`;

    const result = updateCodexExperimentalBearerToken(input, "");

    expect(result).not.toContain("experimental_bearer_token");
    expect(result).not.toContain("sk-old");
  });
});

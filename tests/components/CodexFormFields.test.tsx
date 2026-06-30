import { render, screen } from "@testing-library/react";
import type { ComponentProps, PropsWithChildren } from "react";
import { useForm } from "react-hook-form";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CodexFormFields } from "@/components/providers/forms/CodexFormFields";
import { Form } from "@/components/ui/form";

const modelFetchApiMock = vi.hoisted(() => ({
  fetchModelsForConfig: vi.fn(),
  showFetchModelsError: vi.fn(),
}));

vi.mock("@/lib/api/model-fetch", () => ({
  fetchModelsForConfig: modelFetchApiMock.fetchModelsForConfig,
  showFetchModelsError: modelFetchApiMock.showFetchModelsError,
}));

type CodexFormFieldsProps = ComponentProps<typeof CodexFormFields>;

const FormShell = ({ children }: PropsWithChildren) => {
  const form = useForm();

  return <Form {...form}>{children}</Form>;
};

const renderCodexForm = (overrides: Partial<CodexFormFieldsProps> = {}) => {
  const props: CodexFormFieldsProps = {
    appId: "codex",
    providerId: "custom-codex",
    codexApiKey: "",
    onApiKeyChange: vi.fn(),
    category: "custom",
    shouldShowApiKeyLink: false,
    websiteUrl: "",
    shouldShowSpeedTest: true,
    codexBaseUrl: "https://api.example.com/v1",
    onBaseUrlChange: vi.fn(),
    isFullUrl: false,
    onFullUrlChange: vi.fn(),
    isEndpointModalOpen: false,
    onEndpointModalToggle: vi.fn(),
    onCustomEndpointsChange: vi.fn(),
    autoSelect: true,
    onAutoSelectChange: vi.fn(),
    apiFormat: "openai_responses",
    onApiFormatChange: vi.fn(),
    codexChatReasoning: {},
    onCodexChatReasoningChange: vi.fn(),
    catalogModels: [],
    onCatalogModelsChange: vi.fn(),
    speedTestEndpoints: [],
    customUserAgent: "",
    onCustomUserAgentChange: vi.fn(),
    localProxyHeadersOverride: "",
    onLocalProxyHeadersOverrideChange: vi.fn(),
    localProxyBodyOverride: "",
    onLocalProxyBodyOverrideChange: vi.fn(),
    ...overrides,
  };

  return render(
    <FormShell>
      <CodexFormFields {...props} />
    </FormShell>,
  );
};

describe("CodexFormFields", () => {
  beforeEach(() => {
    modelFetchApiMock.fetchModelsForConfig.mockResolvedValue([]);
    if (!Element.prototype.hasPointerCapture) {
      Element.prototype.hasPointerCapture = () => false;
    }
    if (!Element.prototype.setPointerCapture) {
      Element.prototype.setPointerCapture = () => {};
    }
    if (!Element.prototype.releasePointerCapture) {
      Element.prototype.releasePointerCapture = () => {};
    }
  });

  it("does not auto-expand advanced options for the default Responses format alone", () => {
    renderCodexForm();

    expect(screen.queryByLabelText("自定义 User-Agent")).not.toBeInTheDocument();
    expect(screen.getByText("高级选项")).toBeInTheDocument();
  });

  it("shows Anthropic Messages as a Codex upstream format", () => {
    renderCodexForm({ apiFormat: "anthropic_messages" });

    expect(screen.getByLabelText("自定义 User-Agent")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "上游格式" })).toHaveTextContent(
      "Anthropic Messages",
    );
  });

  it("renders Codex catalog rows that target Anthropic Messages upstream", () => {
    renderCodexForm({
      catalogModels: [
        {
          model: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          provider: "anthropic",
          endpoint: "/v1/messages",
          authHeader: "x-api-key",
        },
      ],
    });

    expect(screen.getByDisplayValue("Claude Sonnet")).toBeInTheDocument();
    expect(screen.getByDisplayValue("claude-sonnet-4-6")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "上游端点" })).toHaveTextContent(
      "/v1/messages",
    );
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps, PropsWithChildren } from "react";
import { useForm } from "react-hook-form";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CodexFormFields } from "@/components/providers/forms/CodexFormFields";
import { Form } from "@/components/ui/form";

const modelFetchApiMock = vi.hoisted(() => ({
  fetchJoycodeModelsForConfig: vi.fn(),
  fetchModelsForConfig: vi.fn(),
  showFetchModelsError: vi.fn(),
  syncJoycodeLoginFromVscode: vi.fn(),
}));
const settingsApiMock = vi.hoisted(() => ({
  openExternal: vi.fn(),
}));

vi.mock("@/lib/api/model-fetch", () => ({
  fetchJoycodeModelsForConfig: modelFetchApiMock.fetchJoycodeModelsForConfig,
  fetchModelsForConfig: modelFetchApiMock.fetchModelsForConfig,
  showFetchModelsError: modelFetchApiMock.showFetchModelsError,
  syncJoycodeLoginFromVscode: modelFetchApiMock.syncJoycodeLoginFromVscode,
}));

vi.mock("@/lib/api", () => ({
  settingsApi: settingsApiMock,
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
    modelFetchApiMock.fetchJoycodeModelsForConfig.mockResolvedValue([]);
    modelFetchApiMock.fetchModelsForConfig.mockResolvedValue([]);
    modelFetchApiMock.syncJoycodeLoginFromVscode.mockResolvedValue({
      userName: "hechengjun.9",
      tenant: "JD",
      loginType: "ERP",
      ptKey: "pt-key",
    });
    settingsApiMock.openExternal.mockResolvedValue(undefined);
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

  it("uses the JoyCode official model list command and merges fetched models into the catalog", async () => {
    const onCatalogModelsChange = vi.fn();
    modelFetchApiMock.fetchJoycodeModelsForConfig.mockResolvedValue([
      {
        id: "GPT-5.3-codex",
        ownedBy: "openai-response",
        displayName: "GPT-5.3-codex",
        upstreamModel: "GPT 5.3-codex",
        provider: "openai",
        endpoint: "/v1/responses",
        authHeader: "bearer",
        responsesMode: "codex",
      },
      {
        id: "Claude-Sonnet-4.6",
        ownedBy: "anthropic",
        displayName: "Claude-Sonnet-4.6",
        upstreamModel: "Claude-Sonnet-4.6-hq",
        provider: "anthropic",
        endpoint: "/v1/messages",
        authHeader: "x-api-key",
      },
    ]);

    renderCodexForm({
      providerId: "joycode",
      codexBaseUrl: "https://joycode-api.jd.com/api/saas/openai/v1",
      codexApiKey: "",
      localProxyHeadersOverride: '{ "ptKey": "pt-key", "tenant": "JD" }',
      onCatalogModelsChange,
    });

    fireEvent.click(
      screen.getByRole("button", { name: /providerForm.fetchModels/ }),
    );

    await waitFor(() => {
      expect(modelFetchApiMock.fetchJoycodeModelsForConfig).toHaveBeenCalledWith(
        '{ "ptKey": "pt-key", "tenant": "JD" }',
      );
      expect(onCatalogModelsChange).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            model: "GPT-5.3-codex",
            upstreamModel: "GPT 5.3-codex",
            endpoint: "/v1/responses",
            responsesMode: "codex",
          }),
          expect.objectContaining({
            model: "Claude-Sonnet-4.6",
            upstreamModel: "Claude-Sonnet-4.6-hq",
            provider: "anthropic",
            endpoint: "/v1/messages",
            authHeader: "x-api-key",
          }),
        ]),
      );
    });
    expect(modelFetchApiMock.fetchModelsForConfig).not.toHaveBeenCalled();
  });

  it("shows a JoyCode login button that opens the official browser login page", () => {
    renderCodexForm({
      providerId: "joycode",
      codexBaseUrl: "https://joycode-api.jd.com/api/saas/openai/v1",
    });

    fireEvent.click(
      screen.getByRole("button", { name: /登录 JoyCode|Login with JoyCode/ }),
    );

    expect(settingsApiMock.openExternal).toHaveBeenCalledWith(
      "http://joycoder.jd.com?login=1&ideAppName=vscode&fromIde=joycode-plugin&redirect=0",
    );
  });

  it("syncs JoyCode login state from the official VS Code extension into header overrides", async () => {
    const onHeadersChange = vi.fn();
    modelFetchApiMock.syncJoycodeLoginFromVscode.mockResolvedValue({
      userName: "hechengjun.9",
      tenant: "JD",
      loginType: "ERP",
      ptKey: "synced-pt-key",
    });

    renderCodexForm({
      providerId: "joycode",
      codexBaseUrl: "https://joycode-api.jd.com/api/saas/openai/v1",
      localProxyHeadersOverride: '{ "X-Provider": "cc-switch" }',
      onLocalProxyHeadersOverrideChange: onHeadersChange,
    });

    fireEvent.click(
      screen.getByRole("button", {
        name: /同步 VS Code 登录态|Sync VS Code Login/,
      }),
    );

    await waitFor(() => {
      expect(modelFetchApiMock.syncJoycodeLoginFromVscode).toHaveBeenCalled();
      expect(onHeadersChange).toHaveBeenCalledWith(
        JSON.stringify(
          {
            "X-Provider": "cc-switch",
            ptKey: "synced-pt-key",
            loginType: "ERP",
            tenant: "JD",
          },
          null,
          2,
        ),
      );
    });
  });
});

import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { codeInspectorPlugin } from "code-inspector-plugin";

export default defineConfig(({ command }) => ({
  root: "src",
  plugins: [
    command === "serve" &&
      codeInspectorPlugin({
        bundler: "vite",
      }),
    react(),
  ].filter(Boolean),
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    strictPort: true,
    // 热重载时减少文件系统监听开销
    watch: {
      usePolling: false,
      interval: 100,
    },
    // 预构建重型依赖，避免首次 HMR 卡顿
    fs: {
      strict: false,
    },
  },
  // 预打包常用依赖，减少按需优化导致的冷启动延迟
  optimizeDeps: {
    include: [
      "react",
      "react-dom",
      "react-router-dom",
      "@tanstack/react-query",
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-accordion",
      "@radix-ui/react-checkbox",
      "@radix-ui/react-collapsible",
      "@radix-ui/react-label",
      "@codemirror/state",
      "@codemirror/view",
      "@codemirror/lang-javascript",
      "@codemirror/lang-json",
      "@codemirror/lang-markdown",
      "@dnd-kit/core",
      "@dnd-kit/sortable",
      "zustand",
    ],
    // 排除不需要预构建的大型包
    exclude: ["@tauri-apps/api"],
  },
  // esbuild 目标设为 esnext 加快转译
  esbuild: {
    target: "esnext",
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  clearScreen: false,
  envPrefix: ["VITE_", "TAURI_"],
}));

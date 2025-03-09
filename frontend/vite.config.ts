import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import { resolve } from 'path';

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  css: {
    modules: {
      localsConvention: "camelCaseOnly"
    }
  },
  server: {
    // Allow all hosts
    host: true,
    cors: true
  },
  // Ensure public directory is properly served
  publicDir: resolve(__dirname, 'public'),
  // Disable type checking is handled in the Dockerfile
});

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind IPv4 loopback explicitly. Node 17+ resolves the default host
    // 'localhost' to ::1 on Windows, so Vite ends up listening on [::1]:3000
    // only -- while uvicorn binds 127.0.0.1:8000. The page is then served over
    // IPv6 and its fetches to http://localhost:8000 try ::1 first, where
    // nothing is listening; that attempt takes ~2s to fail before falling back
    // to IPv4, and when it is abandoned instead the UI shows
    // "Core backend unreachable" (which misreports the cause as CORS).
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  test: {
    // Unit tests only. tests/*.spec.mjs are Playwright scripts that drive a
    // real browser against running APIs; they run via `npm test`, not here.
    include: ['src/**/*.test.{js,jsx}'],
  },
});

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API calls to the Lightspeed Stack backend (see lightspeed-stack.yaml
    // `service.port`) so the UI can call same-origin `/v1/*` paths in dev
    // without hitting CORS.
    proxy: {
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})

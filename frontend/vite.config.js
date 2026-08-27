import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The app always calls a same-origin /api path. In dev this proxy forwards it to the
// local uvicorn; in the container nginx does the same to the `api` service. Keeping
// the browser on one origin means no CORS configuration exists to get wrong.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})

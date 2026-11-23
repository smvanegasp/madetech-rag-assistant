import path from 'path';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    // The app calls /api on its own origin (CloudFront on AWS); locally that's the backend.
    proxy: {
      '/api': 'http://localhost:9481',
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    }
  },
  test: {
    include: ['**/*.test.ts'],
    exclude: ['node_modules/**', 'dist/**'],
  },
});

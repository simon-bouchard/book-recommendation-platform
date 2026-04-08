import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  base: mode === 'production' ? '/static/dist/' : '/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        'search-app': path.resolve(__dirname, 'index.html'),
        'book-app': path.resolve(__dirname, 'book.html'),
        'profile-app': path.resolve(__dirname, 'profile.html'),
        'chat-app': path.resolve(__dirname, 'chat.html'),
        'login-app': path.resolve(__dirname, 'login.html'),
        'signup-app': path.resolve(__dirname, 'signup.html'),
        'home-app': path.resolve(__dirname, 'home.html'),
      },
      output: {
        entryFileNames: 'assets/[name].js',
        assetFileNames: (info) => {
          if (info.name?.endsWith('.css')) return 'assets/styles.css'
          return 'assets/[name].[ext]'
        },
        chunkFileNames: 'assets/chunks/[hash].js',
      },
    },
  },
  server: {
    proxy: {
      '/search/json': 'http://localhost:8000',
      '/search/autocomplete': 'http://localhost:8000',
      '/subjects': 'http://localhost:8000',
      '/book': 'http://localhost:8000',
      '/rating': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/profile': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/login': 'http://localhost:8000',
      '/signup': 'http://localhost:8000',
    },
  },
}))

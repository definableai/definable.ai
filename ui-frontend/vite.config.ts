import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // Define process.env for browser compatibility
    'process.env': {}
  },
  build: {
    // Output as a library for CDN distribution
    lib: {
      entry: 'src/main.tsx',
      name: 'DefinableChat',
      formats: ['es', 'umd'],
      fileName: (format) => `definable-chat.${format}.js`
    },
    rollupOptions: {
      // Externalize deps that shouldn't be bundled
      external: [],
      output: {
        // Provide global variables for UMD build
        globals: {}
      }
    },
    // Generate sourcemaps for debugging
    sourcemap: true,
    // Ensure CSS is extracted
    cssCodeSplit: false
  }
})

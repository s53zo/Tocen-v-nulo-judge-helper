import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'assets',
    emptyOutDir: true,
    sourcemap: false,
    lib: {
      entry: 'src/main.ts',
      formats: ['es'],
      fileName: 'app',
      cssFileName: 'style',
    },
  },
});

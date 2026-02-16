# Epoch — Deep Learning Lab

> **"Code is Light"** — Interactive Deep Learning learning platform

![Epoch](https://img.shields.io/badge/Epoch-Deep%20Learning%20Lab-00e5ff?style=for-the-badge&labelColor=0a0e1a)

A desktop-optimized PWA for learning Deep Learning through interactive code, visualizations and exercises.

## Features

- 🧠 **Neural Roadmap** — Directed graph of 9 learning modules (Tensors → Transformers)
- ⚡ **Split-Screen Lab** — Theory + Code side by side
- 📐 **KaTeX Equations** — Beautiful interactive math rendering
- 🎨 **Neural Diagrams** — Animated SVG network visualizations
- 💻 **Code Reactor** — Monaco Editor with custom neon theme
- 📊 **Tensor Monitor** — Console with heatmap visualizations
- 🔗 **Hover-to-Connect** — Link code variables to theory diagrams
- 📱 **PWA** — Installable as desktop app

## Tech Stack

- React 19 + TypeScript
- Vite 7
- Monaco Editor
- Framer Motion
- KaTeX
- PWA (vite-plugin-pwa)

## Getting Started

```bash
cd epoch-app
npm install
npm run dev
```

## Design Philosophy

**Cyberpunk Minimaliste** — Dark mode first, neon accents (cyan, magenta, orange, green) on deep anthracite backgrounds. The design serves concentration, not decoration.

## License

MIT

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

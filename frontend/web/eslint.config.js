import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  // public/ holds vendored third-party builds (GSAP); linting minified
  // upstream code reports its style, not ours.
  { ignores: ["dist/**", "node_modules/**", "public/**"] },
  {
    files: ["**/*.{js,jsx,mjs}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...js.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // React Compiler rules introduced in eslint-plugin-react-hooks v7. The flagged
      // hooks (useAsync/useAction) were debugged against StrictMode double-invoke;
      // rewriting them to satisfy these is a behavioural change, so they report as
      // warnings until that refactor is done deliberately.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/refs": "warn",
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      // JSX usage is not tracked without eslint-plugin-react; ignore capitalised
      // names so imported components are not reported as unused.
      "no-unused-vars": ["error", { varsIgnorePattern: "^[A-Z_]", argsIgnorePattern: "^_" }],
    },
  },
];

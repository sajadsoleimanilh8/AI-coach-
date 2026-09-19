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
      // React Compiler rules from eslint-plugin-react-hooks v7 stay at the
      // recommended "error" level: the codebase was brought into line with them
      // (render-time state adjustment instead of setState-in-effect), and doing
      // so exposed a real paging loop in PlaybackPanel -- see paging.js.
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      // JSX usage is not tracked without eslint-plugin-react; ignore capitalised
      // names so imported components are not reported as unused.
      "no-unused-vars": ["error", { varsIgnorePattern: "^[A-Z_]", argsIgnorePattern: "^_" }],
    },
  },
];

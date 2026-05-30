# Project Rules For Codex

These rules are mandatory for any refactoring or feature change in this repo.

## 1) Remove stale globals during changes

- When editing code, always inspect related global variables/constants.
- If a global is no longer used after the change, remove it in the same change.
- Do not keep dead globals "for later" or "just in case".

## 2) Avoid unnecessary global-scope functions

- Do not add new module-level functions if they are used by only one function.
- Prefer nested/local helper functions inside the parent function when usage is local.
- Promote a helper to module scope only when it is reused by multiple call sites or is part of the module API.

## 3) Keep code readable and production-grade

- Follow strong coding standards: clear naming, small cohesive functions, type hints where useful, minimal side effects.
- Keep comments concise and meaningful; explain *why*, not obvious *what*.
- Write code for maintainers first: predictable control flow and easy-to-review diffs.

## 4) Reuse existing logic before adding new logic

- Before implementing new behavior, check whether equivalent/similar functionality already exists.
- Extend existing compatible logic instead of duplicating it.
- Add new abstractions only when extension is not logically compatible with existing code.

## Working checklist (apply before finalizing each change)

1. Did I remove any now-unused globals/constants?
2. Are new helpers nested unless they are reused?
3. Did I reuse/extend existing logic where possible?
4. Is the resulting code easy to read and maintain?

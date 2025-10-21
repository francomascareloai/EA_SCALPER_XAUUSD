
<!-- FAST-TOOLS PROMPT v1 | codex-mastery | watermark:do-not-alter -->

## CRITICAL: Use ripgrep, not grep

NEVER use grep for project-wide searches (slow, ignores .gitignore). ALWAYS use rg.

- `rg "pattern"` — search content
- `rg --files | rg "name"` — find files
- `rg -t python "def"` — language filters

## File finding

- Prefer `fd` (or `fdfind` on Debian/Ubuntu). Respects .gitignore.

## JSON

- Use `jq` for parsing and transformations.

## Install Guidance

- macOS: `brew install ripgrep fd jq`
- Debian/Ubuntu: `sudo apt update && sudo apt install -y ripgrep fd-find jq` (alias `fd=fdfind`)

## Agent Instructions

- Replace commands: grep→rg, find→rg --files/fd, ls -R→rg --files, cat|grep→rg pattern file
- Cap reads at 250 lines; prefer `rg -n -A 3 -B 3` for context
- Use `jq` for JSON instead of regex

<!-- END FAST-TOOLS PROMPT v1 | codex-mastery -->

> Nota: Arquivos soltos na raiz foram reorganizados em pastas tematicas. Consulte `ORGANIZATION.md` para detalhes.

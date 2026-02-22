# Paper Finder - Claude Code Compatible

## Setup

Copy this skill to your Claude Code skills directory:
```bash
cp -r skills/paper-finder ~/.claude/skills/paper-finder
```

Add to your project's `CLAUDE.md`:
```markdown
## Skills
- Paper Finder: Use `/paper <keywords>` to find academic papers for PoC.
  See ~/.claude/skills/paper-finder/SKILL.md
```

## Usage in Claude Code

Since Claude Code doesn't have `web_search` or `web_fetch`, use `exec` + `curl`:

### Semantic Scholar Search
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=<keywords>&limit=20&fields=title,authors,year,citationCount,url,externalIds,abstract"
```

### Individual Paper Lookup (more reliable, less rate limiting)
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ArXiv:<arxiv_id>?fields=title,authors,year,citationCount,url,externalIds,abstract"
```

### Papers with Code
```bash
curl -s "https://paperswithcode.com/api/v1/search/?q=<keywords>"
```

## Differences from OpenClaw Version

| Feature | OpenClaw | Claude Code |
|---------|----------|-------------|
| Search API | `web_search` tool | `curl` via `exec` |
| Fallback | `web_search` + `web_fetch` | `curl` + browser search |
| Chat output | Telegram/messaging | Terminal/IDE |
| File save | Same (`docs/papers/`) | Same (`docs/papers/`) |
| Trigger | Auto-detect + `/paper` | `/paper` or ask directly |

## Notes
- Semantic Scholar API rate limit is strict without API key (100 req/5min)
- Individual paper lookups (`/paper/ArXiv:XXXX`) are more reliable than search
- If rate limited (429), wait 60s and retry
- Optional: Get free API key at https://www.semanticscholar.org/product/api#api-key-form

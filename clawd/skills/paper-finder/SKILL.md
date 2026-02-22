---
name: paper-finder
description: "Use when (1) the user message starts with /paper, OR (2) user asks for academic papers, research references, or scholarly citations for PoC, technical decisions, or presentations. Searches Semantic Scholar and Papers with Code for foundational and recent papers with citation counts, URLs, and PoC application points."
---

# Paper Finder

## Overview
Find academic papers that provide scholarly grounding for PoC technical implementations. Returns foundational (원조) + recent papers with URLs, citation counts, and PoC application points.

## When to Use
- `/paper <keywords>` — direct trigger
- User asks for papers/references related to a technology
- Building a PoC deck and needs academic backing
- Wants to understand the research behind a technique

## Search Process

### Step 1: Extract Keywords
Convert user input to English academic keywords.
- Korean input → English translation
- Expand abbreviations (RAG → Retrieval Augmented Generation)
- Add related terms for broader coverage

### Step 2: Search Semantic Scholar API
```
GET https://api.semanticscholar.org/graph/v1/paper/search
  ?query=<keywords>
  &limit=20
  &fields=title,authors,year,citationCount,url,externalIds,abstract
```

### Step 3: Rank Results
- **원조 (Foundational):** Highest citations + older → pick 1
- **최신 (Recent):** Last 2 years + highest citations → pick 2

### Step 4: Enrich with Papers with Code
```
GET https://paperswithcode.com/api/v1/search/?q=<keywords>
```
Add GitHub/code links if available.

### Step 5: Generate Summary
For each paper, produce:
- Title, authors (first author et al.), year
- **Paper URL** (arXiv or Semantic Scholar) — MANDATORY
- Citation count
- One-line summary (Korean)
- Key contribution
- PoC application point
- Code link (if available)

## Output Format

### Chat Output
```
🔬 "<topic>" 관련 논문 3개:

📄 [원조] <Title>
   - <First Author> et al., <Year> | 인용: <count>
   - <URL>
   - 요약: <one-line Korean summary>
   - PoC 적용: <how this applies to the PoC>
   - 💻 코드: <github link if available>

📄 [최신] <Title>
   ...

📄 [최신] <Title>
   ...

## References (덱 복붙용)
[1] <Author> et al. (<Year>). "<Title>" <arXiv/DOI link>

💾 docs/papers/<date>-<topic>.md 저장 완료
```

### Markdown File (docs/papers/YYYY-MM-DD-<topic>.md)
Same content as chat output, saved for reuse. Include full References section for deck copy-paste.

## API Details

### Semantic Scholar API (Primary)
- **Free**, no API key required
- Rate limit: strict (100 req/5min without key)
- Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search`
- Fields: title,authors,year,citationCount,url,externalIds,abstract
- externalIds.ArXiv → `https://arxiv.org/abs/<id>`
- If rate limited (429): wait 60s and retry, or fall back to web search

### Web Search Fallback
- Use `web_search` with `site:arxiv.org OR site:semanticscholar.org <keywords>`
- Then fetch individual paper details from Semantic Scholar by paper ID
- `GET https://api.semanticscholar.org/graph/v1/paper/<paperId>?fields=...`

### Papers with Code (Supplementary)
- **Free**, no API key required
- Endpoint: `https://paperswithcode.com/api/v1/search/?q=<query>`
- Provides: paper + code repository links

## Quick Reference

| Item | Value |
|------|-------|
| Default count | 3 (1 foundational + 2 recent) |
| More papers | User says "더 찾아줘" → search again |
| Language | Keywords in English, summaries in Korean |
| Save location | `docs/papers/YYYY-MM-DD-<topic>.md` |
| URL | Always include arXiv or Semantic Scholar link |

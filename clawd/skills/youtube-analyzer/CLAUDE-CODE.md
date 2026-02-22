# YouTube Analyzer - Claude Code Setup

## Setup
```bash
# Install dependencies
pip install yt-dlp youtube-transcript-api

# Copy skill
cp -r skills/youtube-analyzer ~/.claude/skills/youtube-analyzer
```

Add to `CLAUDE.md`:
```markdown
## Skills
- YouTube Analyzer: Use `/yt <URL>` to extract transcript and analyze YouTube videos.
  See ~/.claude/skills/youtube-analyzer/SKILL.md
```

## Usage (Claude Code)
Works directly with `yt-dlp` since local IP is not blocked:

```bash
# Quick transcript extraction
yt-dlp --write-auto-sub --sub-lang en --skip-download --sub-format vtt -o "/tmp/%(id)s" "https://youtube.com/watch?v=VIDEO_ID"
cat /tmp/VIDEO_ID.en.vtt
```

## Differences from OpenClaw Version

| Feature | OpenClaw (Cloud) | Claude Code (Local) |
|---------|-----------------|-------------------|
| yt-dlp | ❌ Blocked | ✅ Works |
| youtube-transcript-api | ❌ Blocked | ✅ Works |
| Invidious API | ⚠️ Unreliable | ✅ Works |
| Browser fallback | ✅ Available | ✅ Available |

**Claude Code is the recommended environment for this skill.**

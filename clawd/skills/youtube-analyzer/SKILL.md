---
name: youtube-analyzer
description: "Use when (1) the user message starts with /yt, OR (2) user shares a YouTube URL and asks for transcript, summary, or content analysis. Extracts video transcripts and provides structured analysis."
---

# YouTube Analyzer

## Overview
Extract YouTube video transcripts and analyze content. Provides summaries, key points, and technical insights.

## When to Use
- `/yt <YouTube URL>` — direct trigger
- User shares a YouTube link and asks for summary/analysis
- User wants to extract transcript from a video

## Transcript Extraction Methods

### Method 1: OpenClaw Server (web_fetch)
Try `web_fetch` on transcript proxy services:
```
web_fetch: https://inv.nadeko.net/api/v1/captions/<video_id>
```
List available captions, then fetch specific language.

### Method 2: Claude Code / Local (yt-dlp) — RECOMMENDED
```bash
# Install once
pip install yt-dlp

# List available subtitles
yt-dlp --list-subs "<youtube_url>"

# Download auto-generated English subtitles
yt-dlp --write-auto-sub --sub-lang en --skip-download --sub-format vtt -o "/tmp/%(id)s" "<youtube_url>"

# Download manual subtitles (if available, higher quality)
yt-dlp --write-sub --sub-lang en --skip-download --sub-format vtt -o "/tmp/%(id)s" "<youtube_url>"

# Download Korean subtitles
yt-dlp --write-auto-sub --sub-lang ko --skip-download --sub-format vtt -o "/tmp/%(id)s" "<youtube_url>"
```

### Method 3: youtube-transcript-api (Python, local only)
```python
from youtube_transcript_api import YouTubeTranscriptApi

ytt = YouTubeTranscriptApi()
transcript = ytt.fetch("VIDEO_ID")
text = " ".join([s.text for s in transcript.snippets])
```

### Method 4: Browser (OpenClaw or Claude Code)
If API methods fail, use browser automation:
1. Open YouTube video URL
2. Click "..." → "Show transcript"
3. Copy transcript text

## Video ID Extraction
```python
import re
def extract_video_id(url):
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None
```

## VTT → Clean Text
```python
import re
def vtt_to_text(vtt_content):
    """Remove VTT timestamps and metadata, return clean text."""
    lines = vtt_content.split('\n')
    text_lines = []
    for line in lines:
        # Skip timestamps, WEBVTT header, empty lines
        if re.match(r'^\d{2}:\d{2}', line): continue
        if line.startswith('WEBVTT'): continue
        if line.startswith('Kind:'): continue
        if line.startswith('Language:'): continue
        if not line.strip(): continue
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean and clean not in text_lines[-1:]:
            text_lines.append(clean)
    return ' '.join(text_lines)
```

## Analysis Output Format

```
🎬 YouTube 분석: <Video Title>
📎 <URL>
⏱️ <Duration>

## 📝 요약 (3-5문장)
<핵심 내용 요약>

## 🔑 핵심 포인트
1. <point 1>
2. <point 2>
3. <point 3>
...

## 💡 기술적 인사이트 (PoC 관련 시)
- <insight for technical application>

## 📄 전체 트랜스크립트
<full transcript text>

💾 docs/youtube/YYYY-MM-DD-<video-id>.md 저장 완료
```

## Cloud Server Limitations
- AWS/GCP/Azure IP에서는 YouTube가 봇으로 차단
- `yt-dlp`, `youtube-transcript-api` 모두 클라우드에서 차단됨
- **해결책**: 로컬 머신(Claude Code)에서 실행하거나, 프록시 사용

## Quick Reference

| Item | Value |
|------|-------|
| Trigger | `/yt <URL>` or share YouTube link |
| Best method (local) | `yt-dlp` |
| Best method (cloud) | Invidious API or browser |
| Output | Summary + key points + full transcript |
| Save location | `docs/youtube/YYYY-MM-DD-<video-id>.md` |
| Subtitle languages | Auto-detect, prefer manual > auto-generated |

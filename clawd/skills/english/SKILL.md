---
name: english
description: English-Korean translation for business English proficiency improvement. Use when (1) the user message starts with "/eng" prefix, OR (2) continuous translation mode is active (check memory/english-mode.json). Provides literal/direct translations suitable for business communication and professional language learning. Translates Korean to English or English to Korean based on the input language. Supports one-shot mode (/eng text) and continuous mode (/eng start ... /eng end or /eng stop).
---

# English Translation Skill

## Overview

Provides bidirectional translation between English and Korean with a focus on literal/direct translation for business English proficiency improvement. Supports both one-shot and continuous translation modes.

## Translation Guidelines

### Trigger Modes

**Mode 1: One-shot translation**
- Command: `/eng [text to translate]`
- Translates the given text immediately
- Examples:
  - `/eng 저는 AWS에서 일합니다`
  - `/eng I work at AWS`

**Mode 2: Continuous translation**
- Start: `/eng start`
- End: `/eng end` or `/eng stop`
- While active, ALL user messages (except `/eng end` or `/eng stop`) are automatically translated
- State is tracked in `memory/english-mode.json`

### Mode Management

**Before processing any message:**
1. Check if `memory/english-mode.json` exists
2. If it exists and contains `{"active": true}`, continuous mode is ON
3. If mode is ON, translate the user's message (even without `/eng` prefix)
4. If user sends `/eng end` or `/eng stop`, delete the file or set `{"active": false}` and confirm mode is OFF

**Starting continuous mode:**
- User: `/eng start`
- Action: Create `memory/english-mode.json` with `{"active": true}`
- Response: "연속 번역 모드가 시작되었습니다. 이제부터 모든 메시지가 자동으로 번역됩니다. 종료하려면 `/eng end` 또는 `/eng stop`을 입력하세요."

**Ending continuous mode:**
- User: `/eng end` or `/eng stop`
- Action: Delete `memory/english-mode.json` or set `{"active": false}`
- Response: "연속 번역 모드가 종료되었습니다."

### Detection and Direction

**Automatic language detection:**
- Korean input → Translate to English
- English input → Translate to Korean

### Translation Style

**Direct/Literal translation** - prioritized for interview practice:
- Use natural but straightforward phrasing
- Avoid overly poetic or idiomatic interpretations
- Maintain the structure and meaning of the original
- Focus on clarity and accuracy over creativity

### Format

**Single word/phrase:**
```
Input: /eng apple
Output: 사과

Input: /eng 안녕하세요
Output: Hello
```

**Sentence:**
```
Input: /eng I love programming
Output: 나는 프로그래밍을 좋아합니다

Input: /eng 저는 5년간 소프트웨어 엔지니어로 일했습니다
Output: I have worked as a software engineer for 5 years
```

**Multiple sentences or paragraphs:**
Translate each sentence/paragraph and maintain the structure.

## Business English Context

When translating Korean to English for business scenarios:
- Use professional, formal language appropriate for business communication
- Prefer complete sentences over fragments
- Use standard business English terminology
- Maintain a confident, clear, and professional tone
- Focus on clarity and directness suitable for workplace communication

## Response Format

Simply provide the translation directly without excessive explanation:

✅ **Good:**
```
User: /eng 나는 팀과 협력하여 프로젝트를 성공적으로 완료했습니다
Agent: I successfully completed the project by collaborating with the team
```

❌ **Avoid:**
```
The translation of "나는 팀과 협력하여..." is "I successfully completed..."
Let me translate that for you:
Here's the English version:
```

Keep it clean and direct - just provide the translation.

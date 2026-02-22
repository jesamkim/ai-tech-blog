# Skills - Claude Code Compatibility Guide

## Overview

This repo contains skills built for **OpenClaw** (AI agent framework). 
Most skills can also be used with **Claude Code** (Anthropic CLI agent) with minor adjustments.

## Quick Setup for Claude Code

```bash
# Copy all skills to Claude Code's skill directory
cp -r skills/* ~/.claude/skills/

# Add references to your project's CLAUDE.md
```

## Tool Mapping (OpenClaw → Claude Code)

| OpenClaw Tool | Claude Code Equivalent |
|---------------|----------------------|
| `web_search` | `exec` → `curl` (Brave API / direct) |
| `web_fetch` | `exec` → `curl` |
| `message` | N/A (no messaging) |
| `memory_search` | `read` (manual file search) |
| `cron` | N/A (no scheduler) |
| `exec` | `exec` ✅ (same) |
| `read` / `write` / `edit` | ✅ Same |
| `browser` | `browser` ✅ (same) |

## Skills Compatibility

| Skill | Claude Code Ready? | Notes |
|-------|-------------------|-------|
| paper-finder | ⚠️ Needs curl | See `paper-finder/CLAUDE-CODE.md` |
| english | ✅ Ready | Text-only skill |
| brainstorming | ✅ Ready | Process skill, no special tools |
| humanizer | ✅ Ready | Text-only skill |
| writing-plans | ✅ Ready | Process skill |
| test-driven-development | ✅ Ready | Process skill |
| systematic-debugging | ✅ Ready | Process skill |
| verification-before-completion | ✅ Ready | Process skill |
| writing-skills | ✅ Ready | Process skill |
| executing-plans | ✅ Ready | Process skill |
| dispatching-parallel-agents | ⚠️ Limited | Needs `sessions_spawn` equivalent |
| subagent-driven-development | ⚠️ Limited | Needs sub-agent support |

## Per-Skill Notes

Skills with `CLAUDE-CODE.md` have specific adaptation instructions.
Process-only skills (brainstorming, TDD, etc.) work as-is — they describe workflows, not tool calls.

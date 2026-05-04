# Blog Post Task: AgentCore Managed Harness Deep Dive

## Topic
Amazon Bedrock AgentCore Managed Harness — released in preview on April 22, 2026.
Write an in-depth technical blog post in **Korean** (한국어) targeting Korean AWS/AI developers.

## Metadata
- **Slug** (English): `bedrock-agentcore-managed-harness-deep-dive`
- **Date**: `2026-04-26T10:00:00+09:00` (KST, MUST include +09:00)
- **Category**: `AWS AI/ML`
- **Tags**: `AgentCore`, `Bedrock`, `AI Agents`, `Strands Agents`, `Preview`
- **Author**: Jesam Kim

## Sources (verified)
1. AWS ML Blog (official): https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/
2. AWS Whats New: https://aws.amazon.com/about-aws/whats-new/2026/04/agentcore-new-features-to-build-agents-faster/
3. AgentCore CLI GitHub: https://github.com/aws/agentcore-cli
4. ClassMethod deep dive (EN): https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/
5. SiliconANGLE coverage: https://siliconangle.com/2026/04/22/aws-accelerates-ai-agent-development-amazon-bedrock-agentcore/
6. AgentCore docs: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html

## Core Facts (verified, USE THESE)
- Released **April 22, 2026** as **preview**
- Available in **4 regions**: us-west-2 (Oregon), us-east-1 (N. Virginia), eu-central-1 (Frankfurt), ap-southeast-2 (Sydney)
- Three components announced together: **Managed Harness** + **AgentCore CLI** + **AgentCore Skills** (for coding assistants)
- Managed Harness lets you deploy an agent in **3 API calls** by declaring only: `model`, `systemPrompt`, `tools`
- Powered by **Strands Agents** (AWS open source framework) under the hood
- Each session runs in isolated **microVM** with dedicated filesystem + shell
- Session state persisted to durable filesystem — agents can suspend and resume
- Skills for: **Kiro** (already available as Power), **Claude Code**, **Codex**, **Cursor** (plugins coming end of April)
- CLI version: `1.0.0-preview.1`, install via `npm install -g @aws/agentcore@preview`
- Default model (harness): `global.anthropic.claude-sonnet-4-6`
- Supported model providers: Amazon Bedrock, OpenAI, Google Gemini
- Built-in tools: AgentCore Browser, Code Interpreter, Gateway, Remote MCP Server
- Pricing: **no additional charge** for CLI/harness/skills — pay only for underlying resources
- Deployment: CDK supported, Terraform coming soon
- Customer testimonial: VTEX (Rodrigo Moreira, VP of Engineering)

## JSON config example (from ClassMethod, verified)
```json
{
  "name": "MyHarness",
  "model": {
    "provider": "bedrock",
    "modelId": "global.anthropic.claude-sonnet-4-6"
  },
  "tools": [
    {
      "type": "agentcore_code_interpreter",
      "name": "code-interpreter"
    }
  ],
  "skills": []
}
```

## Project structure (from ClassMethod, verified)
```
harnessSample/
├── agentcore/
│   ├── agentcore.json    # Overall project specs
│   ├── aws-targets.json  # Deployment region/account
│   └── cdk/              # CDK stack
└── app/
    └── MyHarness/
        ├── harness.json      # Harness config
        └── system-prompt.md  # System prompt
```

## CLI workflow (verified)
```bash
npm install -g @aws/agentcore@preview
agentcore create      # interactive wizard
agentcore deploy      # CDK-based deployment
agentcore dev         # local dev with web inspector UI
```

## Suggested Structure (6 sections, ~5500 words Korean)
1. **도입 — 에이전트 개발의 90%가 인프라 plumbing이었다** (왜 AgentCore Harness가 필요한가)
2. **Managed Harness 핵심 개념** — harness란 무엇인가, 3개 선언(model/systemPrompt/tools)의 철학
   - [DIAGRAM: Architecture diagram showing traditional agent dev stack vs Managed Harness abstraction layers]
3. **실전 배포 가이드** — `agentcore create` → `agentcore deploy` 전체 흐름, harness.json 구조
4. **내부 아키텍처** — Strands Agents, microVM 세션 격리, 영속 파일시스템, Runtime 관계
   - [DIAGRAM: Session lifecycle showing microVM isolation + persistent filesystem + suspend/resume flow]
5. **언제 써야 하나 — Managed Harness vs Code-defined vs Bedrock Agents** (의사결정 프레임워크)
   - [DIAGRAM: Decision tree / comparison table visualization]
6. **Preview 제약사항과 Production 고려점** — 4개 리전 제약, IAM auth 기본, Observability, 한국 개발자를 위한 팁
7. **마무리** — 핵심 정리 + 다음 단계 (CLI 설치 → 첫 harness 만들어보기)

## Requirements
- **Language**: Korean 한국어 본문
- **Title**: 한국어 제목 + 영문 부제 OK
- **Tone**: Professional but accessible. Jay = AWS Solutions Architect perspective. 관찰자가 아니라 실전 적용자 시점.
- **Length**: ~5000-6500 Korean characters
- **Code examples**: Include harness.json, CLI commands, project structure — all verified above
- **Inline links**: Use `[text](URL)` for specific claims. References at bottom, NO inline `[1]` style.
- **No fabricated metrics**: Only use verified numbers. No made-up "X% faster" claims.
- **Year**: 2026 everywhere. Do NOT write 2024 or 2025.
- **Model names**: Claude Opus 4.6/Sonnet 4.6 (NOT 4.0/4.5). GPT-5.4. Gemini 2.5 Pro.

## Diagrams (svg-diagram skill)
Use `/Workshop/ai-skills/my-skills/svg-diagram/SKILL.md` and `~/.claude/skills/moai-domain-uiux/SKILL.md`:
1. **Stack comparison**: traditional agent infra (framework + storage + auth + deployment) vs Managed Harness abstraction
2. **Session lifecycle**: microVM + persistent FS + suspend/resume
3. **Decision tree**: when to use Managed Harness vs Code-defined vs Bedrock Agents

All diagrams MUST:
- Watermark bottom-right: `jesamkim.github.io` (alpha 0.7)
- Save as PNG to `/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/`
- Image paths in post: `/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/<name>.png`
- Alt text + caption with source

## Cover image (sd35l)
After post is generated, create cover:
```python
ssh command in SKILL.md — use sd35l with profile2 in us-west-2
prompt: "Photorealistic cinematic wide-angle digital illustration of AI agent orchestration infrastructure, abstract neural network nodes connected to cloud service icons, dark navy background with orange and teal accents, dramatic lighting, ultra detailed 8k, tech blog cover style, no text"
negative_prompt: "text, watermark, logo, blurry, low quality, cartoon, anime, people faces"
```
Save to: `/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/cover.png`

Add to frontmatter (right after `draft: false`):
```yaml
cover:
  image: "/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/cover.png"
  alt: "Amazon Bedrock AgentCore Managed Harness"
  relative: false
```

## QA Before Finishing
- [ ] All `**bold**` → `<strong>` in CJK context
- [ ] No inline `[1]` citations, only `[text](URL)` style
- [ ] References section at bottom with all 6 sources
- [ ] All years = 2026
- [ ] Humanizer pass applied (no "획기적인", "혁신적인", "다양한" AI-ese)
- [ ] File saved to `/Workshop/yan/ai-tech-blog/content/posts/2026-04-26-bedrock-agentcore-managed-harness-deep-dive.md`
- [ ] Cover image generated + frontmatter includes cover block
- [ ] At least 2 SVG diagrams generated with watermark
- [ ] Category = AWS AI/ML (existing)
- [ ] No customer names used (Jay cant mention Samsung)

## AWS Profile
Use `AWS_PROFILE=profile2` for all Bedrock calls (default profile in use by Jay).

## Do NOT
- Do NOT git commit or push — Jay will review first
- Do NOT publish — draft: false OK, but Jay reviews before deploy
- Do NOT invent customer names, quotes, or metrics
- Do NOT use Korean URL slugs

Save the final post and report:
1. File path
2. Word count (Korean chars)
3. Number of diagrams generated
4. Cover image path
5. Any facts you were unsure about

## ADDENDUM: Screenshot Images (Jay request)

Download and use these AWS official screenshots (safest, most authoritative source):

### Image 1: Harness Invoke Demo
- Source URL: `https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2026/04/21/AgentCore-DevEx-Capabilities_Screenshot_HarnessInvokeDemo_Blog_1200w_REVIEW.jpg`
- Save to: `/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/harness-invoke-demo.jpg`
- Use in: section on Managed Harness experience / CLI demo
- Caption: `*출처: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)*`

### Image 2: CLI sample
- Source URL: `https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2026/04/21/AgentCore-DevEx-Capabilities_Screenshot_cli-sample_Blog_1500w_REVIEW.jpg`
- Save to: `/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/cli-sample.jpg`
- Use in: CLI workflow section
- Caption: `*출처: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)*`

### Download commands
```bash
cd /Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/
curl -L "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2026/04/21/AgentCore-DevEx-Capabilities_Screenshot_HarnessInvokeDemo_Blog_1200w_REVIEW.jpg" -o harness-invoke-demo.jpg
curl -L "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2026/04/21/AgentCore-DevEx-Capabilities_Screenshot_cli-sample_Blog_1500w_REVIEW.jpg" -o cli-sample.jpg
```

### Image embedding format
```markdown
![Managed Harness invoke demo](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/harness-invoke-demo.jpg)
*Managed Harness CLI invoke 결과. 출처: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)*
```

### Additional option: ClassMethod console screenshots (use SPARINGLY, only if adding unique value)
If you need a console UI screenshot (Quick create harness etc.), you MAY download 1-2 from ClassMethod with clear attribution:
- ClassMethod article URL: https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/
- Console sidebar Harness Preview menu: `https://devio2024-media.developers.io/image/upload/f_auto/q_auto/v1776900210/2026/04/23/ivv1sxhkt23mhtdqazjw.png`
- Quick create playground: `https://devio2024-media.developers.io/image/upload/f_auto/q_auto/v1776900230/2026/04/23/rlkztlwjwmybp5dkmpoh.png`
- If used, caption: `*출처: [DevelopersIO - ClassMethod](https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/)*`

**Priority**: AWS official images first (always OK). ClassMethod screenshots second (only if they add unique value not covered by AWS images, max 2 images, clear attribution).

Total images target: 4-6 (2 AWS official + 2-3 SVG diagrams + optionally 1-2 ClassMethod with attribution + 1 cover)

## ADDENDUM 2: Cover Image Style Override (Jay request)

**IMPORTANT**: Override the previous cover image prompt. Use **anime/animation style** instead of photorealistic cinematic.

### New cover prompt (use this, ignore previous photorealistic one)
```
prompt: "Anime style digital illustration, modern Japanese animation aesthetic, AI agent orchestration infrastructure concept, glowing cloud service nodes connected by flowing data streams, futuristic tech workspace with holographic interfaces, dark navy and deep purple background with cyan and orange accent lights, dramatic lighting, cel-shaded, studio ghibli meets cyberpunk, high quality anime art, detailed background, no text, no characters, 16:9 aspect ratio"
negative_prompt: "photorealistic, photograph, real people, faces, text, watermark, logo, blurry, low quality, 3d render, ugly, deformed"
```

### sd35l command (use profile2, us-west-2)
```bash
ssh 2026-poc has already what you need. Execute on 2026-poc itself since you ARE on 2026-poc:
python3 -c "
import boto3, json, base64
b = boto3.Session(profile_name=\"profile2\").client(\"bedrock-runtime\", region_name=\"us-west-2\")
body = json.dumps({
    \"prompt\": \"Anime style digital illustration, modern Japanese animation aesthetic, AI agent orchestration infrastructure concept, glowing cloud service nodes connected by flowing data streams, futuristic tech workspace with holographic interfaces, dark navy and deep purple background with cyan and orange accent lights, dramatic lighting, cel-shaded, studio ghibli meets cyberpunk, high quality anime art, detailed background, no text, no characters\",
    \"negative_prompt\": \"photorealistic, photograph, real people, faces, text, watermark, logo, blurry, low quality, 3d render, ugly, deformed\",
    \"aspect_ratio\": \"16:9\",
    \"output_format\": \"png\"
})
r = b.invoke_model(modelId=\"stability.sd3-5-large-v1:0\", body=body, contentType=\"application/json\", accept=\"application/json\")
img = base64.b64decode(json.loads(r[\"body\"].read())[\"images\"][0])
open(\"/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/cover.png\", \"wb\").write(img)
print(\"Cover saved\")
"
```

### Style policy (going forward — from this post onwards)
- Default cover style: **anime/animation** (not photorealistic)
- Adjust anime sub-style per topic mood when needed (cyberpunk, ghibli, shonen tech, etc.)
- Negative prompt MUST include: `photorealistic, real people, faces` to prevent mixing

### Frontmatter cover block (same as before)
```yaml
cover:
  image: "/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/cover.png"
  alt: "Amazon Bedrock AgentCore Managed Harness"
  relative: false
```

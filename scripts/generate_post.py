#!/usr/bin/env python3
"""Bedrock Claude로 한국어 블로그 포스트 생성 (섹션별 분할 생성)"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import boto3
import yaml
from botocore.config import Config as BotoConfig

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
TEMPLATES_DIR = BASE_DIR / "templates"
HUGO_DIR = BASE_DIR / "hugo-site"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_post")


def load_config() -> dict:
    with open(SCRIPTS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")[:60]


def call_bedrock(prompt: str, config: dict, max_tokens: int = 2048) -> str:
    """Bedrock Claude API 호출 (스트리밍)"""
    bedrock_cfg = config.get("bedrock", {})
    boto_config = BotoConfig(read_timeout=300, connect_timeout=10, retries={"max_attempts": 2})
    client = boto3.client("bedrock-runtime", region_name=bedrock_cfg.get("region", "us-west-2"), config=boto_config)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": bedrock_cfg.get("temperature", 0.7),
        "messages": [{"role": "user", "content": prompt}],
    }
    response = client.invoke_model_with_response_stream(
        modelId=bedrock_cfg.get("model_id", "global.anthropic.claude-sonnet-4-20250514-v1:0"),
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    chunks = []
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            chunks.append(chunk["delta"]["text"])
    return "".join(chunks)


# ── 섹션별 프롬프트 ──────────────────────────────────────────

def prompt_outline(topic: str, sources: list, config: dict) -> str:
    gen_cfg = config.get("generation", {})
    source_text = ""
    for s in sources[:5]:
        source_text += f"- [{s['title']}]({s['link']}): {s.get('summary', '')[:150]}\n"

    return f"""당신은 AI/ML 전문 블로거 '{gen_cfg.get("author", "Jesam Kim")}'입니다.
다음 주제의 블로그 포스트 **아웃라인**을 작성하세요.

## 주제: {topic}

## 참고 소스
{source_text}

## 출력
1. Hugo front matter (YAML)
2. 섹션 제목 목록 (## 헤더) — 보통 4~6개 섹션
3. 각 섹션의 핵심 포인트 (불릿 2~3개)

형식:
---
title: "제목"
date: {datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")}
draft: false
author: "{gen_cfg.get("author", "Jesam Kim")}"
description: "한 줄 설명"
categories:
  - "카테고리"
tags:
  - "태그1"
  - "태그2"
ShowToc: true
TocOpen: true
---

## 섹션1 제목
- 포인트1
- 포인트2

## 섹션2 제목
...

아웃라인만 작성하세요. 본문은 작성하지 마세요."""


def prompt_section(topic: str, section_title: str, section_points: str, prev_sections: str, config: dict) -> str:
    gen_cfg = config.get("generation", {})
    return f"""당신은 AI/ML 전문 블로거 '{gen_cfg.get("author", "Jesam Kim")}'입니다.
블로그 포스트 "{topic}"의 한 섹션을 작성합니다.

## 이 섹션
**제목**: {section_title}
**핵심 포인트**:
{section_points}

## 이전 섹션 요약 (맥락 유지용)
{prev_sections if prev_sections else "(첫 섹션입니다)"}

## 요구사항
- 한국어로 작성, 기술 용어는 영어 병기
- 코드 예시 포함 가능 (Python)
- 400~800자 분량
- Mermaid 다이어그램이 유용하면 ```mermaid 블록 포함
- 섹션 제목(## 헤더)부터 시작
- 이 섹션의 내용만 작성 (다른 섹션 내용 쓰지 마세요)"""


def prompt_humanize(section_text: str) -> str:
    return f"""당신은 전문 편집자입니다. 아래 한국어 기술 블로그 글에서 AI가 쓴 티가 나는 부분을 자연스럽게 교정하세요.

## 교정 규칙
1. "~의 중요성을 강조합니다", "~에 있어 핵심적인 역할을 합니다" 같은 과장 제거
2. "Additionally", "Furthermore" 등 AI 특유 접속사 → 자연스러운 한국어 연결
3. "~를 보여줍니다(showcasing)", "~를 반영합니다(reflecting)" 같은 -ing 번역체 제거
4. 볼드(**) 남발 줄이기 — 정말 중요한 곳만
5. 이모지 제거 (코드 블록 내부 제외)
6. "~의 태피스트리", "~의 랜드스케이프" 같은 AI 은유 제거
7. 3개씩 나열하는 패턴 줄이기 (rule of three)
8. em dash(—) 과다 사용 줄이기
9. 문장 길이와 구조 다양하게 — 짧은 문장, 긴 문장 섞기
10. 필자의 의견이나 경험이 자연스럽게 들어가면 좋음 ("개인적으로", "실제로 써보면")
11. 코드 블록, Mermaid 블록, 이미지 링크, front matter는 절대 수정하지 마세요
12. 기술 용어의 영어 병기는 유지

## 입력
{section_text}

## 출력
교정된 글만 출력하세요. 설명이나 변경사항 요약은 불필요합니다."""


def prompt_references(topic: str, sources: list) -> str:
    source_text = ""
    for s in sources[:5]:
        source_text += f"- [{s['title']}]({s['link']})\n"
    return f"""블로그 포스트 "{topic}"의 References 섹션을 작성하세요.

## 사용된 소스
{source_text}

## 요구사항
- ## References 헤더로 시작
- 번호 매기기
- URL 포함
- 추가로 관련된 중요 논문/자료가 있으면 2~3개 추가
- 한국어로 작성

## URL 규칙 (반드시 준수)
- AWS 서비스 링크는 docs.aws.amazon.com 공식 문서를 우선 사용 (영문판 고정, 로케일 리다이렉트 방지)
- aws.amazon.com/서비스명/ 형태는 서비스 종료 시 리다이렉트될 수 있으므로 피할 것
- 레퍼런스 설명과 실제 URL 종류가 일치해야 함 (예: "공식 문서"라고 쓰면서 블로그 링크를 달지 말 것)
  - docs.aws.amazon.com → "공식 문서"
  - aws.amazon.com/blogs/ → "AWS Blog"
  - arxiv.org → "논문"
  - github.com → "GitHub"
- 실제로 존재하는 URL만 사용. 추측으로 URL을 만들지 말 것"""


# ── 아웃라인 파싱 ────────────────────────────────────────────

def parse_outline(outline_text: str) -> tuple:
    """아웃라인에서 front matter + 섹션 목록 추출"""
    # front matter 추출
    fm_match = re.search(r"(---\n.*?---)", outline_text, re.DOTALL)
    front_matter = fm_match.group(1) if fm_match else ""

    # 섹션 추출
    sections = []
    current_section = None
    current_points = []

    for line in outline_text.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections.append({"title": current_section, "points": "\n".join(current_points)})
            current_section = line
            current_points = []
        elif line.strip().startswith("-") and current_section:
            current_points.append(line)

    if current_section:
        sections.append({"title": current_section, "points": "\n".join(current_points)})

    return front_matter, sections


# ── 메인 ─────────────────────────────────────────────────────

def extract_mermaid_blocks(content: str) -> list:
    pattern = r"```mermaid\n(.*?)```"
    return re.findall(pattern, content, re.DOTALL)


def generate_post(topic: str, sources: list = None, config: dict = None) -> Path:
    """섹션별 분할 생성"""
    if config is None:
        config = load_config()
    if sources is None:
        sources = []

    # Step 1: 아웃라인 생성
    logger.info("Step 1/N: 아웃라인 생성")
    outline_prompt = prompt_outline(topic, sources, config)
    outline_text = call_bedrock(outline_prompt, config, max_tokens=1024)
    logger.info("아웃라인 완료 (%d자)", len(outline_text))

    front_matter, sections = parse_outline(outline_text)
    if not sections:
        logger.error("아웃라인 파싱 실패, 원문:\n%s", outline_text[:500])
        raise ValueError("아웃라인에서 섹션을 추출할 수 없습니다")

    logger.info("섹션 %d개: %s", len(sections), [s["title"][:30] for s in sections])

    # Step 2: 섹션별 본문 생성
    generated_sections = []
    for i, sec in enumerate(sections):
        logger.info("Step %d/%d: %s", i + 2, len(sections) + 2, sec["title"][:40])
        prev_summary = ""
        if generated_sections:
            # 이전 섹션들의 첫 2줄만 요약으로 전달
            for gs in generated_sections[-2:]:
                lines = gs.strip().split("\n")
                prev_summary += lines[0] + "\n" + (lines[1] if len(lines) > 1 else "") + "\n\n"

        sec_prompt = prompt_section(topic, sec["title"], sec["points"], prev_summary, config)
        sec_text = call_bedrock(sec_prompt, config, max_tokens=1500)
        generated_sections.append(sec_text)
        logger.info("  → %d자 생성", len(sec_text))

    # Step 3: References
    logger.info("Step %d/%d: References", len(sections) + 2, len(sections) + 2)
    ref_text = call_bedrock(prompt_references(topic, sources), config, max_tokens=512)

    # Step 4: Humanize (섹션별 교정)
    humanized_sections = []
    for i, sec_text in enumerate(generated_sections):
        logger.info("Humanize %d/%d", i + 1, len(generated_sections))
        humanized = call_bedrock(prompt_humanize(sec_text), config, max_tokens=2000)
        humanized_sections.append(humanized)
        logger.info("  → %d자 → %d자", len(sec_text), len(humanized))

    # Step 5: 조립
    content = front_matter + "\n\n" + "\n\n".join(humanized_sections) + "\n\n" + ref_text

    # 다이어그램 처리
    mermaid_blocks = extract_mermaid_blocks(content)
    if mermaid_blocks:
        try:
            from generate_diagram import generate_diagram
            today = datetime.now().strftime("%Y-%m-%d")
            for i, block in enumerate(mermaid_blocks):
                img_path = generate_diagram(block, output_name=f"diagram-{i+1}", date_str=today)
                if img_path:
                    old = f"```mermaid\n{block}```"
                    new = f"![다이어그램 {i+1}]({img_path})"
                    content = content.replace(old, new)
        except Exception as e:
            logger.warning("다이어그램 생성 실패: %s", e)

    # 저장
    title_match = re.search(r'^title:\s*"(.+?)"', content, re.MULTILINE)
    title = title_match.group(1) if title_match else topic
    slug = slugify(title)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{today}-{slug}.md"

    posts_dir = HUGO_DIR / "content" / "posts"
    posts_dir.mkdir(parents=True, exist_ok=True)
    output_path = posts_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("✅ 포스트 저장: %s (%d자, %d섹션)", output_path, len(content), len(sections))
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="블로그 포스트 생성")
    parser.add_argument("--topic", required=True, help="포스트 주제")
    parser.add_argument("--sources-file", help="소스 JSON 파일 경로")
    args = parser.parse_args()

    sources = []
    if args.sources_file:
        with open(args.sources_file) as f:
            sources = json.load(f)

    output = generate_post(args.topic, sources)
    print(f"생성 완료: {output}")


if __name__ == "__main__":
    main()

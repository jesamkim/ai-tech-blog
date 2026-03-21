#!/usr/bin/env python3
"""Generate podcast audio from blog post.

Pipeline:
  1. Read blog markdown
  2. Use Bedrock Claude to generate a natural conversational script (~15 min)
  3. Apply EN→KR pronunciation mapping
  4. Generate audio via Amazon Polly
"""

import boto3
import subprocess
import tempfile
import os
import re
import sys
import json

# ── Config ──────────────────────────────────────────────────────────
POST_PATH = "/Workshop/yan/ai-tech-blog/content/posts/2026-03-20-physical-ai-guide-vlm-vla-world-model.md"
OUTPUT_PATH = "/Workshop/yan/ai-tech-blog/static/audio/2026-03-20-physical-ai.mp3"
SCRIPT_CACHE = "/tmp/podcast_script_cache.json"  # Cache generated script

TARGET_MINUTES = 13  # Target ~13 min (Korean ~250 chars/min for natural speech)
TARGET_CHARS = TARGET_MINUTES * 250 * 2  # ~6500 chars total (both speakers)

# ── Step 1: Generate conversational script via Bedrock ──────────────

SCRIPT_PROMPT = """당신은 AI 기술 팟캐스트의 대본 작가입니다.

아래 블로그 글을 바탕으로, 두 명의 진행자가 자연스럽게 대화하는 팟캐스트 대본을 작성하세요.

## 진행자
- **서연** (호스트): 질문하고, 요약하고, 청취자 관점에서 쉽게 풀어주는 역할
- **지혜** (전문가): 기술적 내용을 설명하되, 일상적 비유를 많이 사용

## 핵심 규칙
1. **완전한 구어체**: "~입니다" 대신 "~거든요", "~인데요", "~잖아요" 등 실제 대화체
2. **URL, 괄호, 참고문헌 절대 금지**: "(2024)" 같은 연도 괄호도 금지. "2024년에 나온" 식으로 풀어쓰기
3. **영어 약어는 한글 발음으로**: VLM→브이엘엠, AI→에이아이, GPU→지피유 등
4. **기술 용어는 쉬운 비유와 함께**: "임베딩이라는 건... 쉽게 말하면 숫자로 된 요약이에요"
5. **자연스러운 맞장구**: "아~", "그렇죠", "맞아요", "오 그거 재밌네요" 등
6. **핵심만 압축**: 블로그 전체를 다 다루지 말고, 가장 흥미로운 포인트 위주로
7. **리스너 친화적**: "여러분도 챗지피티 써보셨죠?" 같은 공감 포인트

## 분량
- 총 대사 합쳐서 약 {target_chars}자 (한국어 기준 약 {target_min}분 분량)
- 인트로 + 본문 5~7개 토픽 + 아웃트로
- 한 사람의 대사는 최대 200자 (길면 나눠서)

## 출력 형식 (JSON)
```json
[
  {{"speaker": "서연", "text": "안녕하세요! 오늘은..."}},
  {{"speaker": "지혜", "text": "네, 오늘 주제가..."}},
  ...
]
```

반드시 유효한 JSON 배열만 출력하세요. 다른 텍스트 없이 JSON만.

---

## 블로그 원문:

{content}
"""


def generate_script_via_llm(blog_content):
    """Use Bedrock Claude to create a conversational podcast script."""
    
    # Check cache first
    if os.path.exists(SCRIPT_CACHE):
        print("📄 Using cached script from", SCRIPT_CACHE)
        with open(SCRIPT_CACHE, 'r') as f:
            return json.load(f)
    
    print("🤖 Generating conversational script via Bedrock Claude...")
    
    bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    prompt = SCRIPT_PROMPT.format(
        content=blog_content[:12000],  # Trim to avoid token limits
        target_chars=TARGET_CHARS,
        target_min=TARGET_MINUTES,
    )
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    })
    
    response = bedrock.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-6",
        body=body,
    )
    
    result = json.loads(response['body'].read())
    text = result['content'][0]['text']
    
    # Extract JSON from response (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    
    script = json.loads(text)
    
    # Cache it
    with open(SCRIPT_CACHE, 'w') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(script)} dialogue turns")
    total_chars = sum(len(d['text']) for d in script)
    print(f"   Total chars: {total_chars} (target: {TARGET_CHARS})")
    print(f"   Est. duration: {total_chars / 250 / 2:.1f} min")
    
    return script


# ── Step 2: EN→KR Pronunciation ────────────────────────────────────

COMPOUNDS = {
    "NVIDIA": "엔비디아", "OpenAI": "오픈에이아이", "DeepMind": "딥마인드",
    "Hugging Face": "허깅페이스", "Physical Intelligence": "피지컬 인텔리전스",
    "Google DeepMind": "구글 딥마인드",
    "ChatGPT": "챗지피티", "GPT-4V": "지피티 포브이", "GPT-4": "지피티 포",
    "GR00T": "그루트", "V-JEPA": "브이제파", "JEPA": "제파",
    "CLIP": "클립", "LoRA": "로라", "LiDAR": "라이다",
    "RT-2": "알티투", "RT-1": "알티원",
    "OpenVLA": "오픈브이엘에이", "LLaVA": "라바",
    "PaLM-E": "팜이", "PaLI": "팔리",
    "ResNet": "레즈넷", "LLaMA": "라마", "ViT": "빗",
    "Isaac Sim": "아이작 심", "Isaac": "아이작",
    "Omniverse": "옴니버스", "Cosmos": "코스모스",
    "Jetson Thor": "젯슨 토르", "Jetson": "젯슨",
    "Physical AI": "피지컬 에이아이",
    "World Model": "월드모델", "World Models": "월드모델",
    "Sim-to-Real": "심투리얼", "sim-to-real": "심투리얼",
    "End-to-End": "엔드투엔드", "end-to-end": "엔드투엔드",
    "Fine-tuning": "파인튜닝", "fine-tuning": "파인튜닝",
    "Flow Matching": "플로우 매칭",
    "Dual-system": "듀얼시스템", "dual-system": "듀얼시스템",
    "Embodied Reasoning": "임바디드 리즈닝",
    "Vision-Language-Action": "비전 랭귀지 액션",
    "Vision-Language Model": "비전 랭귀지 모델",
    "Yann LeCun": "얀 르쿤", "LeCun": "르쿤",
    "Jensen Huang": "젠슨 황",
    "SoC": "에스오씨", "GitHub": "깃허브",
    "Waymo": "웨이모", "Gemini": "제미나이",
    "Stanford": "스탠포드", "Google": "구글", "Meta": "메타",
    "Newton": "뉴턴", "Figure": "피규어",
    "pre-trained": "프리트레인드", "few-shot": "퓨샷", "zero-shot": "제로샷",
}

LETTER_KO = {
    'A': '에이', 'B': '비', 'C': '씨', 'D': '디', 'E': '이',
    'F': '에프', 'G': '지', 'H': '에이치', 'I': '아이',
    'J': '제이', 'K': '케이', 'L': '엘', 'M': '엠', 'N': '엔',
    'O': '오', 'P': '피', 'Q': '큐', 'R': '알', 'S': '에스',
    'T': '티', 'U': '유', 'V': '브이', 'W': '더블유', 'X': '엑스',
    'Y': '와이', 'Z': '지',
}

ENGLISH_KO = {
    "physical": "피지컬", "digital": "디지털", "virtual": "버추얼",
    "model": "모델", "models": "모델", "modeling": "모델링",
    "vision": "비전", "language": "랭귀지",
    "action": "액션", "actions": "액션",
    "token": "토큰", "tokens": "토큰",
    "encoder": "인코더", "decoder": "디코더",
    "transformer": "트랜스포머", "projector": "프로젝터",
    "parameter": "파라미터", "dataset": "데이터셋",
    "benchmark": "벤치마크", "pipeline": "파이프라인",
    "architecture": "아키텍처", "framework": "프레임워크",
    "platform": "플랫폼", "interface": "인터페이스",
    "module": "모듈", "layer": "레이어", "layers": "레이어",
    "feature": "피처", "embedding": "임베딩",
    "diffusion": "디퓨전", "generative": "제너레이티브",
    "inference": "인퍼런스", "training": "트레이닝",
    "learning": "러닝", "policy": "폴리시",
    "simulation": "시뮬레이션", "simulator": "시뮬레이터",
    "rendering": "렌더링", "perception": "퍼셉션",
    "planning": "플래닝", "control": "컨트롤",
    "navigation": "내비게이션", "manipulation": "매니풀레이션",
    "locomotion": "로코모션", "grasping": "그래스핑",
    "sensor": "센서", "safety": "세이프티",
    "robot": "로봇", "robots": "로봇", "robotics": "로보틱스",
    "humanoid": "휴머노이드", "autonomous": "오토노머스",
    "embodied": "임바디드", "embodiment": "임바디먼트",
    "data": "데이터", "image": "이미지", "video": "비디오",
    "pixel": "픽셀", "frame": "프레임", "sample": "샘플",
    "scale": "스케일", "scaling": "스케일링",
    "hardware": "하드웨어", "software": "소프트웨어",
    "feedback": "피드백", "output": "아웃풋", "input": "인풋",
    "backbone": "백본", "baseline": "베이스라인",
    "stack": "스택", "chip": "칩", "demo": "데모",
    "mobile": "모바일", "real-time": "리얼타임",
    "open-source": "오픈소스",
    "foundation": "파운데이션", "attention": "어텐션",
    "contrastive": "컨트래스티브", "reasoning": "리즈닝",
    "system": "시스템", "flow": "플로우",
    "source": "소스", "review": "리뷰", "web": "웹",
    "knowledge": "날리지", "general": "제너럴",
    "space": "스페이스", "state": "스테이트",
    "dynamics": "다이나믹스", "physics": "피직스",
    "machine": "머신", "intelligence": "인텔리전스",
    "task": "태스크", "tasks": "태스크",
    "performance": "퍼포먼스", "domain": "도메인",
    "latency": "레이턴시", "randomization": "랜도마이제이션",
}


def _expand_caps(m):
    return ''.join(LETTER_KO.get(c, c) for c in m.group(0))


def preprocess_for_tts(text):
    """3-layer EN→KR pronunciation."""
    # Also strip any remaining markdown/URL artifacts
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\([^)]*https?://[^)]*\)', '', text)
    text = text.replace('(', '').replace(')', '')  # Remove all parentheses
    
    # Layer 1: Compounds
    for eng, kor in sorted(COMPOUNDS.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(eng, kor)
    
    # Layer 2: Uppercase abbreviations
    text = re.sub(r'(?<![A-Za-z])[A-Z]{2,}(?![a-z])', _expand_caps, text)
    
    # Layer 3: English words
    def _replace(m):
        w = m.group(0)
        lower = w.lower()
        return ENGLISH_KO.get(lower, w)
    text = re.sub(r'(?<![A-Za-z가-힣])[A-Za-z]{2,}(?![A-Za-z])', _replace, text)
    
    return text.strip()


def detect_remaining_english(parts):
    IGNORE = {
        'et', 'al', 'vs', 'for', 'and', 'the', 'to', 'into', 'an', 'of', 'in',
        'on', 'or', 'by', 'is', 'it', 'at', 'as', 'be', 'if', 'so', 'no', 'do',
        'Hz', 'ms', 'io', 'km', 'An', 'mm', 'cm',
    }
    remaining = {}
    for idx, (voice, text) in enumerate(parts):
        words = re.findall(r'(?<![A-Za-z가-힣])[A-Za-z]{2,}(?![A-Za-z])', text)
        for w in words:
            if w not in IGNORE:
                remaining.setdefault(w, []).append(idx)
    return remaining


# ── Step 3: Audio generation ────────────────────────────────────────

def generate_audio(parts, output_path):
    polly = boto3.client('polly', region_name='ap-northeast-2')
    temp_files = []

    for idx, (voice, text) in enumerate(parts):
        print(f"  [{idx+1}/{len(parts)}] {voice}: {text[:60]}...")
        
        voice_id = "Seoyeon" if voice == "seoyeon" else "Jihye"
        engine = "generative" if voice == "seoyeon" else "neural"
        ssml = f'<speak>{text}<break time="400ms"/></speak>'

        try:
            resp = polly.synthesize_speech(
                Text=ssml, TextType='ssml', OutputFormat='mp3',
                VoiceId=voice_id, Engine=engine, SampleRate='24000')
            tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir='/tmp')
            tmp.write(resp['AudioStream'].read())
            tmp.close()
            temp_files.append(tmp.name)
        except Exception as e:
            print(f"  ERROR: {e}")
            try:
                resp = polly.synthesize_speech(
                    Text=ssml, TextType='ssml', OutputFormat='mp3',
                    VoiceId='Seoyeon', Engine='neural', SampleRate='24000')
                tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir='/tmp')
                tmp.write(resp['AudioStream'].read())
                tmp.close()
                temp_files.append(tmp.name)
            except Exception as e2:
                print(f"  RETRY FAILED: {e2}")

    print(f"\nGenerated {len(temp_files)} audio segments")

    concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for tf in temp_files:
        concat_list.write(f"file '{tf}'\n")
    concat_list.close()

    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list.name,
        '-codec:a', 'libmp3lame', '-b:a', '128k',
        output_path
    ], check=True, capture_output=True)

    for tf in temp_files:
        os.unlink(tf)
    os.unlink(concat_list.name)

    size = os.path.getsize(output_path)
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', output_path],
        capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f"\n✅ Done! Output: {output_path}")
    print(f"   Size: {size / 1024 / 1024:.1f} MB")
    print(f"   Duration: {duration/60:.1f} minutes")


# ── Main ────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    no_cache = "--no-cache" in sys.argv
    
    if no_cache and os.path.exists(SCRIPT_CACHE):
        os.unlink(SCRIPT_CACHE)
        print("🗑️  Cleared script cache")

    # Read blog post
    with open(POST_PATH, 'r') as f:
        blog_content = f.read()
    
    # Step 1: Generate conversational script via LLM
    script = generate_script_via_llm(blog_content)
    
    # Step 2: Convert to Polly format with pronunciation fixes
    speaker_map = {"서연": "seoyeon", "지혜": "jihye"}
    parts = []
    for turn in script:
        speaker = speaker_map.get(turn["speaker"], "jihye")
        text = preprocess_for_tts(turn["text"])
        if text:
            parts.append((speaker, text))
    
    print(f"\nTotal parts: {len(parts)}")
    total_chars = sum(len(t) for _, t in parts)
    print(f"Total chars: {total_chars} (est. {total_chars/250/2:.1f} min)")
    
    # Post-check remaining English
    remaining = detect_remaining_english(parts)
    if remaining:
        print(f"\n⚠️  {len(remaining)} English words remain:")
        for w, idxs in sorted(remaining.items(), key=lambda x: -len(x[1])):
            print(f"  '{w}' (×{len(idxs)})")
    else:
        print("\n✅ No remaining English words!")
    
    if dry_run:
        print("\n[DRY RUN] Sample lines:")
        for idx, (v, t) in enumerate(parts[:20]):
            print(f"  [{idx}] {v}: {t[:100]}")
        return
    
    # Step 3: Generate audio
    generate_audio(parts, OUTPUT_PATH)


if __name__ == "__main__":
    main()

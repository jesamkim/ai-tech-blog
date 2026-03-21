#!/usr/bin/env python3
"""Generate podcast audio from blog post using Amazon Polly.

3-layer pronunciation pipeline:
  Layer 1: Compound terms (longest-first exact match)
  Layer 2: Uppercase abbreviations → letter-by-letter (AI→에이아이)
  Layer 3: Common English words → Korean pronunciation
  + Post-check: detect remaining English words and report
"""

import boto3
import subprocess
import tempfile
import os
import re
import sys

# ── Config ──────────────────────────────────────────────────────────
POST_PATH = "/Workshop/yan/ai-tech-blog/content/posts/2026-03-20-physical-ai-guide-vlm-vla-world-model.md"
OUTPUT_PATH = "/Workshop/yan/ai-tech-blog/static/audio/2026-03-20-physical-ai.mp3"

# ── Layer 1: Compound / proper nouns (exact match, longest first) ──
COMPOUNDS = {
    # Brands / orgs
    "NVIDIA": "엔비디아", "OpenAI": "오픈에이아이", "DeepMind": "딥마인드",
    "Hugging Face": "허깅페이스", "Physical Intelligence": "피지컬 인텔리전스",
    "Google DeepMind": "구글 딥마인드",
    # Models / products
    "ChatGPT": "챗지피티", "GPT-4V": "지피티 포브이", "GPT-4": "지피티 포",
    "GR00T": "그루트", "V-JEPA": "브이제파", "JEPA": "제파",
    "CLIP": "클립", "LoRA": "로라", "LiDAR": "라이다",
    "RT-2": "알티투", "RT-1": "알티원",
    "OpenVLA": "오픈브이엘에이", "LLaVA": "라바",
    "ASIMOV": "아시모프",
    "PaLM-E": "팜이", "PaLI": "팔리",
    "ResNet": "레즈넷", "LLaMA": "라마", "ViT": "빗",
    "arXiv": "아카이브",
    # NVIDIA stack
    "Isaac Sim": "아이작 심", "Isaac": "아이작",
    "Omniverse": "옴니버스", "Cosmos": "코스모스",
    "Jetson Thor": "젯슨 토르", "Jetson": "젯슨",
    # Concepts (multi-word)
    "Physical AI": "피지컬 에이아이",
    "World Model": "월드모델", "World Models": "월드모델",
    "Sim-to-Real": "심투리얼", "sim-to-real": "심투리얼",
    "End-to-End": "엔드투엔드", "end-to-end": "엔드투엔드",
    "Fine-tuning": "파인튜닝", "fine-tuning": "파인튜닝", "fine-tune": "파인튠",
    "Flow Matching": "플로우 매칭",
    "Dual-system": "듀얼시스템", "dual-system": "듀얼시스템", "Dual-System": "듀얼시스템",
    "Embodied Reasoning": "임바디드 리즈닝",
    "Action Expert": "액션 엑스퍼트",
    "Vision Encoder": "비전 인코더",
    "Vision-Language-Action": "비전 랭귀지 액션",
    "Vision-Language Model": "비전 랭귀지 모델",
    "Vision-Language": "비전 랭귀지",
    "System 1": "시스템 원", "System 2": "시스템 투",
    "Open X-Embodiment": "오픈 엑스 임바디먼트",
    "pre-trained": "프리트레인드", "Pre-trained": "프리트레인드",
    "Pre-training": "프리트레이닝", "pre-training": "프리트레이닝",
    "few-shot": "퓨샷", "zero-shot": "제로샷",
    "contact-rich": "컨택트리치",
    # People
    "Yann LeCun": "얀 르쿤", "LeCun": "르쿤",
    "Jensen Huang": "젠슨 황",
    # Misc
    "SoC": "에스오씨",
    "github": "깃허브", "GitHub": "깃허브",
}

# ── Layer 2: Uppercase abbreviation letter-by-letter ────────────────
LETTER_KO = {
    'A': '에이', 'B': '비', 'C': '씨', 'D': '디', 'E': '이',
    'F': '에프', 'G': '지', 'H': '에이치', 'I': '아이',
    'J': '제이', 'K': '케이', 'L': '엘', 'M': '엠', 'N': '엔',
    'O': '오', 'P': '피', 'Q': '큐', 'R': '알', 'S': '에스',
    'T': '티', 'U': '유', 'V': '브이', 'W': '더블유', 'X': '엑스',
    'Y': '와이', 'Z': '지',
}

def _expand_caps(m):
    return ''.join(LETTER_KO.get(c, c) for c in m.group(0))

# ── Layer 3: Common English words → Korean pronunciation ────────────
# Add words as they appear in blog posts. Case-insensitive matching.
ENGLISH_KO = {
    # Tech general
    "physical": "피지컬", "digital": "디지털", "virtual": "버추얼",
    "model": "모델", "models": "모델", "modeling": "모델링",
    "vision": "비전", "visual": "비주얼",
    "language": "랭귀지",
    "action": "액션", "actions": "액션",
    "token": "토큰", "tokens": "토큰", "tokenization": "토크나이제이션",
    "encoder": "인코더", "decoder": "디코더",
    "transformer": "트랜스포머", "transformers": "트랜스포머",
    "projector": "프로젝터", "projection": "프로젝션",
    "resampler": "리샘플러",
    "parameter": "파라미터", "parameters": "파라미터",
    "dataset": "데이터셋", "datasets": "데이터셋",
    "benchmark": "벤치마크", "benchmarks": "벤치마크",
    "pipeline": "파이프라인", "pipelines": "파이프라인",
    "architecture": "아키텍처",
    "framework": "프레임워크", "frameworks": "프레임워크",
    "platform": "플랫폼",
    "interface": "인터페이스",
    "module": "모듈", "modules": "모듈",
    "layer": "레이어", "layers": "레이어",
    "feature": "피처", "features": "피처",
    "vector": "벡터", "vectors": "벡터",
    "embedding": "임베딩", "embeddings": "임베딩",
    "latent": "레이턴트",
    "diffusion": "디퓨전",
    "generative": "제너레이티브",
    "inference": "인퍼런스",
    "training": "트레이닝", "train": "트레인",
    "learning": "러닝",
    "policy": "폴리시", "policies": "폴리시",
    "reward": "리워드",
    "simulation": "시뮬레이션", "simulator": "시뮬레이터",
    "synthesis": "신시시스",
    "rendering": "렌더링", "render": "렌더",
    "perception": "퍼셉션",
    "planning": "플래닝",
    "control": "컨트롤", "controller": "컨트롤러",
    "navigation": "내비게이션",
    "manipulation": "매니풀레이션",
    "locomotion": "로코모션",
    "dexterous": "덱스터러스",
    "grasping": "그래스핑", "grasp": "그래스프",
    "object": "오브젝트", "objects": "오브젝트",
    "scene": "씬", "scenes": "씬",
    "sensor": "센서", "sensors": "센서",
    "safety": "세이프티",
    "robust": "로버스트", "robustness": "로버스트니스",
    "scalable": "스케일러블", "scalability": "스케일러빌리티",
    "deploy": "디플로이", "deployment": "디플로이먼트",
    "real-time": "리얼타임", "realtime": "리얼타임",
    "open-source": "오픈소스", "opensource": "오픈소스",
    "feedback": "피드백",
    "output": "아웃풋", "input": "인풋",
    "downstream": "다운스트림", "upstream": "업스트림",
    "backbone": "백본",
    "baseline": "베이스라인",
    "state-of-the-art": "스테이트오브디아트",
    "end-effector": "엔드이펙터",
    # Robotics
    "robot": "로봇", "robots": "로봇", "robotics": "로보틱스",
    "humanoid": "휴머노이드", "humanoids": "휴머노이드",
    "autonomous": "오토노머스", "autonomy": "오토노미",
    "embodied": "임바디드", "embodiment": "임바디먼트",
    "actuator": "액추에이터", "actuators": "액추에이터",
    "torque": "토크",
    "gripper": "그리퍼",
    "workspace": "워크스페이스",
    # Companies / proper nouns (single word, case-sensitive handled separately)
    "google": "구글", "meta": "메타", "stanford": "스탠포드",
    "waymo": "웨이모", "gemini": "제미나이",
    "newton": "뉴턴", "figure": "피규어",
    "prismatic": "프리즈매틱",
    "surgical": "서지컬",
    # Common adjectives/nouns in tech text
    "single": "싱글", "multiple": "멀티플",
    "complex": "컴플렉스", "simple": "심플",
    "novel": "노벨",
    "domain": "도메인", "domains": "도메인",
    "task": "태스크", "tasks": "태스크",
    "performance": "퍼포먼스",
    "approach": "어프로치",
    "method": "메소드", "methods": "메소드",
    "technique": "테크닉", "techniques": "테크닉",
    "strategy": "스트래티지",
    "data": "데이터",
    "image": "이미지", "images": "이미지",
    "video": "비디오", "videos": "비디오",
    "pixel": "픽셀", "pixels": "픽셀",
    "frame": "프레임", "frames": "프레임",
    "sample": "샘플", "samples": "샘플", "sampling": "샘플링",
    "batch": "배치",
    "scale": "스케일", "scaling": "스케일링",
    "gap": "갭",
    "stack": "스택",
    "chip": "칩", "chips": "칩",
    "hardware": "하드웨어", "software": "소프트웨어",
    "transfer": "트랜스퍼",
    "zero": "제로",
    "demo": "데모",
    "abstract": "앱스트랙트",
    "representation": "리프레젠테이션", "representations": "리프레젠테이션",
    "masking": "마스킹", "masked": "마스킹된",
    "predict": "프리딕트", "prediction": "프리딕션",
    "commercially": "커머셜리", "commercial": "커머셜",
    "viable": "바이어블",
    # Additional from dry-run
    "pi": "파이",
    "labs": "랩스",
    "world": "월드",
    "foundation": "파운데이션",
    "physics": "피직스",
    "cross": "크로스",
    "attention": "어텐션",
    "survey": "서베이",
    "state": "스테이트",
    "joint": "조인트",
    "predictive": "프리딕티브",
    "space": "스페이스",
    "augmented": "오그먼티드",
    "machine": "머신",
    "intelligence": "인텔리전스",
    "physically": "피지컬리",
    "plausible": "플로저블",
    "rigid": "리지드",
    "body": "바디",
    "dynamics": "다이나믹스",
    "reasoning": "리즈닝",
    "technologies": "테크놀로지스",
    "agility": "어질리티",
    "digit": "디짓",
    "amazon": "아마존",
    "randomization": "랜도마이제이션",
    "system": "시스템",
    "identification": "아이덴티피케이션",
    "check": "체크",
    "description": "디스크립션",
    "bringing": "브링잉",
    "flow": "플로우",
    "general": "제너럴",
    "source": "소스",
    "concepts": "컨셉츠",
    "progress": "프로그레스",
    "applications": "어플리케이션즈",
    "challenges": "챌린지스",
    "review": "리뷰",
    "web": "웹",
    "knowledge": "날리지",
    "robotic": "로보틱",
    "generalist": "제너럴리스트",
    "seed": "시드",
    "round": "라운드",
    "explained": "익스플레인드",
    "mobile": "모바일",
    "manipulator": "매니풀레이터",
    "latency": "레이턴시",
    "linear": "리니어",
    "former": "포머",
    "perceiver": "퍼시버",
    "ordinary": "오디너리",
    "differential": "디퍼런셜",
    "equation": "이퀘이션",
    "contrastive": "컨트래스티브",
    "shot": "샷",
    "large": "라지",
    "assistant": "어시스턴트",
    "open": "오픈",
    "vla": "브이엘에이",
}

# ── Preprocessing pipeline ──────────────────────────────────────────

def preprocess_for_tts(text):
    """3-layer English→Korean pronunciation conversion."""
    
    # Layer 1: Compound terms (longest first for greedy match)
    for eng, kor in sorted(COMPOUNDS.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(eng, kor)
    
    # Layer 2: Remaining uppercase abbreviations (2+ caps not inside English word)
    text = re.sub(r'(?<![A-Za-z])[A-Z]{2,}(?![a-z])', _expand_caps, text)
    
    # Layer 3: Common English words → Korean (case-insensitive)
    def _replace_word(m):
        word = m.group(0)
        lower = word.lower()
        if lower in ENGLISH_KO:
            return ENGLISH_KO[lower]
        return word
    
    # Match English words (2+ letters, not already converted to Korean)
    text = re.sub(r'(?<![A-Za-z가-힣])[A-Za-z]{2,}(?![A-Za-z])', _replace_word, text)
    
    return text


def detect_remaining_english(parts):
    """Post-check: find English words that survived all 3 layers."""
    # Skip short function words and common reference-only terms
    IGNORE = {
        'et', 'al', 'vs', 'for', 'and', 'the', 'to', 'into', 'an', 'of', 'in',
        'on', 'or', 'by', 'is', 'it', 'at', 'as', 'be', 'if', 'so', 'no', 'do',
        'Hz', 'ms', 'io', 'km', 'mm', 'cm', 'kg', 'gb', 'mb', 'An',
        'Kim', 'Brohan', 'Sapkota',  # author names
    }
    remaining = {}
    for idx, (voice, text) in enumerate(parts):
        words = re.findall(r'(?<![A-Za-z가-힣])[A-Za-z]{2,}(?![A-Za-z])', text)
        for w in words:
            if w in IGNORE:
                continue
            if w not in remaining:
                remaining[w] = []
            remaining[w].append(idx)
    return remaining


# ── Markdown → plain text ───────────────────────────────────────────

def strip_markdown(content):
    content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{<.*?>\}\}', '', content)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'#{1,6}\s*', '', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    content = re.sub(r'`[^`]+`', '', content)
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'\|[-:]+\|[-:|\s]+\|', '', content)
    content = re.sub(r'^\s*\|.*\|\s*$',
                     lambda m: m.group().replace('|', ', ').strip(', '),
                     content, flags=re.MULTILINE)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


# ── Build podcast script ────────────────────────────────────────────

def build_script(paragraphs):
    parts = []
    parts.append(("seoyeon",
        "안녕하세요, 에이아이 기술 블로그 팟캐스트에 오신 것을 환영합니다. "
        "오늘은 피지컬 에이아이에 대해 깊이 있게 다뤄보겠습니다. "
        "브이엘엠에서 브이엘에이, 그리고 월드모델까지, "
        "물리 세계에서 행동하는 에이아이의 모든 것을 알아보겠습니다."))
    parts.append(("jihye",
        "네, 좋은 주제입니다. 피지컬 에이아이는 최근 엔비디아 지티씨에서도 "
        "핵심 키워드로 등장할 만큼 뜨거운 분야인데요, 차근차근 살펴보겠습니다."))

    transitions = [
        "그렇군요. 그러면 다음 내용도 이어서 설명해주시겠어요?",
        "흥미롭네요. 이 부분도 좀 더 자세히 들어볼까요?",
        "잘 이해했습니다. 계속해서 다음 부분으로 넘어가볼까요?",
        "좋은 설명이었습니다. 그 다음은 어떤 내용인가요?",
    ]

    for i, para in enumerate(paragraphs):
        para = para.replace('\n', ' ').strip()
        if len(para) < 30:
            continue
        para = preprocess_for_tts(para)
        if len(para) > 800:
            para = para[:800] + "."

        if i % 3 == 0:
            if i > 0:
                parts.append(("seoyeon", transitions[i % len(transitions)]))
            parts.append(("jihye", para))
        elif i % 3 == 1:
            parts.append(("jihye", para))
        else:
            parts.append(("seoyeon", f"아, 그러니까 정리하면 이런 거죠. {para[:200]}"))
            if len(para) > 200:
                parts.append(("jihye", f"네 맞습니다. 좀 더 보충하면, {para[200:]}"))

    parts.append(("seoyeon",
        "오늘도 정말 알찬 내용이었습니다. 피지컬 에이아이, 브이엘엠, 브이엘에이, "
        "월드모델의 개념과 최신 동향까지 잘 정리해주셔서 감사합니다."))
    parts.append(("jihye",
        "감사합니다. 피지컬 에이아이는 앞으로 로봇, 자율주행, 제조업 등 "
        "다양한 분야에서 혁신을 이끌 핵심 기술입니다. 오늘 내용이 도움이 되셨길 바랍니다."))
    parts.append(("seoyeon", "네, 그럼 다음 에피소드에서 또 만나겠습니다. 감사합니다!"))
    return parts


# ── Audio generation ────────────────────────────────────────────────

def generate_audio(parts, output_path):
    polly = boto3.client('polly', region_name='ap-northeast-2')
    temp_files = []

    for idx, (voice, text) in enumerate(parts):
        print(f"  [{idx+1}/{len(parts)}] {voice}: {text[:60]}...")
        
        voice_id = "Seoyeon" if voice == "seoyeon" else "Jihye"
        engine = "generative" if voice == "seoyeon" else "neural"
        ssml = f'<speak>{text}<break time="500ms"/></speak>'

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
            if engine == "generative":
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

    # Merge
    concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for tf in temp_files:
        concat_list.write(f"file '{tf}'\n")
    concat_list.close()

    print("Merging audio segments...")
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
    print(f"\nDone! Output: {output_path}")
    print(f"Size: {size / 1024 / 1024:.1f} MB")
    print(f"Duration: {duration/60:.1f} minutes")


# ── Main ────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv

    with open(POST_PATH, 'r') as f:
        content = f.read()

    content = strip_markdown(content)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 20]
    parts = build_script(paragraphs)

    print(f"Total script parts: {len(parts)}")

    # Post-check: detect remaining English
    remaining = detect_remaining_english(parts)
    if remaining:
        print(f"\n⚠️  {len(remaining)} English words still remain after conversion:")
        for word, indices in sorted(remaining.items(), key=lambda x: -len(x[1])):
            print(f"  '{word}' (×{len(indices)}) — add to ENGLISH_KO or COMPOUNDS")
        print()
    else:
        print("\n✅ No remaining English words detected!\n")

    if dry_run:
        print("[DRY RUN] Skipping audio generation.")
        # Print a few sample lines
        for idx, (v, t) in enumerate(parts[:15]):
            print(f"  [{idx}] {v}: {t[:100]}")
        return

    generate_audio(parts, OUTPUT_PATH)


if __name__ == "__main__":
    main()

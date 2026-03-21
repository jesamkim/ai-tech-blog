#!/usr/bin/env python3
"""Generate conversational podcast script via Bedrock (streaming)."""
import boto3, json, re, subprocess

SCRIPT_CACHE = '/tmp/podcast_script_cache.json'

result = subprocess.run(
    ['ssh', '2026-poc', 'cat /Workshop/yan/ai-tech-blog/content/posts/2026-03-20-physical-ai-guide-vlm-vla-world-model.md'],
    capture_output=True, text=True, timeout=15
)
content = result.stdout[:10000]

TARGET_MINUTES = 13
TARGET_CHARS = TARGET_MINUTES * 250 * 2

prompt = f"""당신은 AI 기술 팟캐스트의 대본 작가입니다.

아래 블로그 글을 바탕으로, 두 명의 진행자가 자연스럽게 대화하는 팟캐스트 대본을 작성하세요.

## 진행자
- 서연 (호스트): 질문하고, 요약하고, 청취자 관점에서 쉽게 풀어주는 역할
- 지혜 (전문가): 기술적 내용을 설명하되, 일상적 비유를 많이 사용

## 핵심 규칙
1. 완전한 구어체: ~입니다 대신 ~거든요, ~인데요, ~잖아요 등 실제 대화체
2. URL, 괄호, 참고문헌 절대 금지. (2024) 같은 연도 괄호도 금지. '2024년에 나온' 식으로 풀어쓰기
3. 영어 약어는 한글 발음으로 표기: VLM은 브이엘엠, AI는 에이아이, GPU는 지피유
4. 기술 용어는 쉬운 비유와 함께 설명
5. 자연스러운 맞장구: 아~, 그렇죠, 맞아요, 오 그거 재밌네요 등
6. 핵심만 압축: 가장 흥미로운 포인트 위주
7. 리스너 친화적: '여러분도 챗지피티 써보셨죠?' 같은 공감 포인트
8. 대사 길이: 한 사람의 대사는 최대 150자

## 분량
- 총 대사 합쳐서 약 {TARGET_CHARS}자 (약 {TARGET_MINUTES}분 분량)
- 인트로 + 본문 5~6개 토픽 + 아웃트로

## 출력 형식
반드시 JSON 배열만 출력. 다른 텍스트 없이.
[
  {{"speaker": "서연", "text": "안녕하세요!..."}},
  {{"speaker": "지혜", "text": "네, 오늘..."}}
]

## 블로그 원문:
{content}
"""

print("Calling Bedrock (streaming)...")
bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')

resp = bedrock.invoke_model_with_response_stream(
    modelId='global.anthropic.claude-sonnet-4-6',
    body=json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 8000,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
    }),
)

chunks = []
for event in resp['body']:
    chunk = json.loads(event['chunk']['bytes'])
    if chunk.get('type') == 'content_block_delta':
        text = chunk['delta'].get('text', '')
        chunks.append(text)
        if len(chunks) % 50 == 0:
            print(f"  ...received {sum(len(c) for c in chunks)} chars")

full_text = ''.join(chunks).strip()
print(f"Total received: {len(full_text)} chars")

if full_text.startswith('```'):
    full_text = re.sub(r'^```(?:json)?\s*', '', full_text)
    full_text = re.sub(r'\s*```$', '', full_text)

script = json.loads(full_text)
with open(SCRIPT_CACHE, 'w') as f:
    json.dump(script, f, ensure_ascii=False, indent=2)

total = sum(len(d['text']) for d in script)
print(f"\nDone: {len(script)} turns, {total} chars, est {total/250/2:.1f} min")
print(f"Saved: {SCRIPT_CACHE}")

print("\n--- Preview (first 10) ---")
for d in script[:10]:
    print(f"[{d['speaker']}] {d['text'][:90]}")

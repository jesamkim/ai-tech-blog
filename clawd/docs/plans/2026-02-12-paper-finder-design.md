# Paper Finder Skill Design

## 개요
- **이름:** `paper-finder`
- **목적:** PoC 기술 구현의 학술적 근거가 되는 논문 검색 및 요약
- **사용 시점:** PoC 시작 전(A) + 진행 중(B) 핵심, 완료 후(C)도 지원

## 트리거
- `/paper <키워드>` - 직접 호출
- 자연어 대화에서 논문 필요 시 자동 인식

## 검색 소스
1. **Semantic Scholar API** (무료, 인용 수/초록/관련 논문)
2. **Papers with Code** (코드 구현 있는 논문)

## 검색 메커니즘
1. 키워드 추출: 한국어 → 영어 학술 키워드 변환
2. Semantic Scholar API 검색 (20개 후보)
3. 랭킹: 원조 논문 1개 (인용 수↑ + 연도↓) + 최신 2개 (2년 내 + 인용 수↑)
4. Papers with Code 보강 (코드 링크)
5. 요약 생성 (한국어, abstract 기반)

## 기본 출력
- 3개 논문 (원조 1 + 최신 2), 추가 요청 가능

## 출력 항목 (논문당)
- 제목, 저자, 연도
- 논문 URL (arXiv/DOI) **필수**
- 인용 수
- 한줄 요약 (한국어)
- 핵심 기여 (Key Contribution)
- PoC 적용 포인트
- 코드 링크 (있으면)

## 출력 형태
1. 채팅으로 즉시 요약 전달
2. `docs/papers/YYYY-MM-DD-<주제>.md` 파일 저장
3. 덱 복붙용 Reference 형식 포함

## 출력 예시 (채팅)
```
🔬 "RAG 건설 문서 검색" 관련 논문 3개:

📄 [원조] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
   - Lewis et al., 2020 | 인용: 5,200+
   - https://arxiv.org/abs/2005.11401
   - 요약: 검색과 생성을 결합한 RAG 패턴 최초 제안
   - PoC 적용: Knowledge Base → Retriever → Generator 파이프라인 근거
   - 💻 코드: github.com/huggingface/transformers

📄 [최신] Self-RAG: Learning to Retrieve, Generate, and Critique
   - Asai et al., 2023 | 인용: 800+
   - https://arxiv.org/abs/2310.11511
   - 요약: 모델이 검색 필요 여부를 스스로 판단
   - PoC 적용: 정확도 개선 시 Self-RAG 도입 근거

📄 [최신] CRAG: Corrective Retrieval Augmented Generation
   - Yan et al., 2024 | 인용: 350+
   - https://arxiv.org/abs/2401.15884
   - 요약: 검색 결과 품질 평가 후 웹 검색으로 보완
   - PoC 적용: 검색 품질 보장 메커니즘 근거

💾 docs/papers/2026-02-12-rag-construction.md 저장 완료
```

## 마크다운 파일 포맷
```markdown
# [주제] 관련 논문

## 검색일: YYYY-MM-DD
## 키워드: ...

### 📄 [원조] 논문 제목
- **저자:** ...
- **연도:** ...
- **인용:** ...
- **URL:** ...
- **요약:** ...
- **핵심 기여:** ...
- **PoC 적용:** ...
- **코드:** ...

(반복)

## References (덱 복붙용)
[1] Lewis, P. et al. (2020). "Retrieval-Augmented Generation..." arXiv:2005.11401
```

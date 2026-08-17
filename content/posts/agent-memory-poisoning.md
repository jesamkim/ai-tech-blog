---
title: "악성 데이터 한 줄이 AI 에이전트의 기억에 영구히 남는다: Memory Poisoning 방어 설계"
date: 2026-08-16T10:00:00+09:00
draft: false
categories: ["논문 리뷰"]
tags: ["Memory Poisoning", "LLM Agent", "AI Security", "Prompt Injection", "Agentic AI", "OWASP"]
author: "Jesam Kim"
description: "지속 메모리를 가진 에이전트는 한 번의 악성 write가 세션 경계를 넘어 남는 공격 표면을 갖게 됩니다. 최근 연구 세 편과 공개 보안 시연 한 건을 근거로 memory poisoning을 설계와 운영의 문제로 다시 읽어봅니다."
cover:
  image: "/ai-tech-blog/images/agent-memory-poisoning/cover.png"
  alt: "에이전트 메모리를 노리는 memory poisoning 공격과 방어 설계"
  relative: false
---

지난 몇 년간 LLM 애플리케이션의 보안 논의는 대체로 prompt injection에 집중되어 있었습니다. 신뢰되지 않은 텍스트가 모델의 지시 사항인 척 끼어드는 문제입니다. 그런데 이 논의의 전제 하나가 조용히 바뀌었습니다. 초기의 LLM 호출은 요청 하나가 끝나면 그 세션의 상태도 함께 사라지는 stateless 구조였지만, 지금의 에이전트는 대화·경험·선호를 세션 경계 너머로 저장하는 지속 메모리를 기본으로 갖추고 있습니다.

이 전환이 만든 것은 새로운 기능이 아니라 새로운 공격 표면입니다. Prompt injection의 payload는 응답이 끝나면 컨텍스트와 함께 사라집니다. 하지만 그 payload가 메모리 쓰기 한 번을 성공시키면, 원본 텍스트가 사라진 뒤에도 저장된 항목은 남습니다. 이 글에서는 이것을 "payload는 휘발되지만 쓰기 결과는 잔존한다"는 성질로 부르겠습니다. memory poisoning을 prompt injection과 따로 다뤄야 하는 이유가 여기에 있습니다.

이 글은 세 편의 연구와 한 건의 공개 보안 시연을 근거로 삼습니다. 메모리 쓰기 경로를 체계적으로 분류한 Dash et al.의 연구, 쿼리만으로 메모리를 오염시키는 MINJA, 그리고 MINJA 방식을 전자건강기록 에이전트 환경에서 다시 실험한 후속 연구가 논문 쪽이고, ChatGPT 메모리를 대상으로 한 SpAIware 시연이 나머지 한 건입니다. 이들을 종합해 memory poisoning을 아키텍처 소개가 아니라 설계·운영 원칙의 문제로 정리합니다.

## payload는 사라져도 메모리 쓰기는 남는다

Prompt injection 방어는 대체로 "이번 요청에 들어온 텍스트를 신뢰할 것인가"라는 단일 시점의 판단입니다. 입력을 검증하고, 시스템 프롬프트와 사용자 입력을 구분하고, 도구 호출 전에 확인을 거치는 식입니다. 이 판단이 틀리더라도 피해는 그 세션 안에 머무는 경우가 많습니다.

메모리가 끼어들면 이 경계가 흐려집니다. 오염된 항목 하나가 메모리에 들어가고 나면, 이후 세션에서 그 항목이 실제로 검색(retrieve)되는 경우 에이전트는 그것을 사실이나 선호나 절차로 취급합니다. 검색 결과에 들어오지 않으면 아무 일도 일어나지 않지만, 들어오는 순간부터는 공격자가 추가로 개입하지 않아도 에이전트가 스스로 그 항목을 불러와 사용합니다. 공격자가 계속 붙어 있어야 하는 prompt injection과 갈라지는 지점이 이것입니다.

자기강화는 여기서 한 단계 더 나아간 문제인데, 모든 메모리 아키텍처에 해당하지는 않습니다. 과거 경험을 절차로 일반화해 저장하거나 자율적으로 skill 파일을 개선하는 설계에서는, 오류 없이 실행된 단계가 검증된 절차로 취급되면서 오염된 판단이 반복 실행을 거치며 더 다듬어질 수 있습니다. Dash et al.은 이 성질을 자기개선 루프가 증폭 장치로 작동하는 취약점으로 따로 분류하고, 정적인 메모리 구조에는 대응물이 없다고 명시합니다. 지속성이 memory poisoning 전반의 성질이라면, 자기강화는 경험을 절차로 승격시키는 아키텍처에 한정된 추가 위험으로 읽는 편이 정확합니다.

![신뢰되지 않은 입력이 memory write 채널을 통과해 지속 메모리에 저장되고, 이후 세션의 검색 결과에 포함되면 행동을 오염시키며, 경험을 절차로 일반화하는 아키텍처에서는 그 결과가 다시 저장되어 강화되는 흐름을 보여주는 다이어그램](/ai-tech-blog/images/agent-memory-poisoning/diagram-1-attack-loop.png)

*공격 흐름과 자기강화 루프. Dash et al.의 위협 모델과 취약점 분류를 재구성한 그림입니다. ⑤번 경로는 경험을 절차로 일반화하는 아키텍처에서만 성립합니다.*

## 구조적으로 뜯어보기: 4개 채널과 9개 취약점

Huawei Canada와 워털루 대학교 연구팀이 2026년 6월 공개한 <a href="https://arxiv.org/abs/2606.04329" target="_blank" rel="noopener">"From Untrusted Input to Trusted Memory"</a> 논문은 memory poisoning을 개별 공격 사례가 아니라 시스템적으로 분석합니다. 이 논문이 정리한 memory write 채널은 네 가지입니다. 사용자나 도구가 명시적으로 저장을 지시하는 경로, 시스템 프롬프트 설계가 특정 정보를 저장하도록 유도하는 경로, 긴 대화를 압축할 때 요약 과정에 끼어드는 경로, 과거 행동의 결과를 절차로 일반화해 저장하는 경로입니다.

이 네 채널을 가능하게 하는 구조적 취약점을 저자들은 아홉 가지로 분류하고, 이를 다시 모델 능력 층위, 시스템 프롬프트 층위, 에이전트 아키텍처 층위로 나눕니다. 이 구분이 중요한 이유는 취약점이 한 곳에 모여 있지 않다는 점을 보여주기 때문입니다. 모델을 더 안전한 버전으로 바꾸는 것만으로는 시스템 프롬프트 설계나 아키텍처에 뚫려 있는 경로를 막을 수 없습니다.

논문은 여기서 얻은 taxonomy를 여섯 개 공격 유형으로 정리하고, 이를 자동으로 재현할 수 있는 <a href="https://arxiv.org/abs/2606.04329" target="_blank" rel="noopener">MPBench 벤치마크</a>를 함께 제안합니다. 저자들이 강조하는 결론은 두 가지입니다. 하나는 메모리를 더 적극적으로 쓰고 불러오는 방식으로 설계된 에이전트일수록 취약점 표면이 커진다는 점입니다. 다른 하나는 기존 prompt injection 방어 장치를 그대로 얹어서는 memory poisoning을 막지 못한다는 점입니다. 두 위협이 겹치는 지점도 있지만, 후자는 "쓰기가 성공한 뒤"라는 시점을 방어 대상으로 명시적으로 잡지 않으면 소용이 없습니다.

## 이상적인 조건과 실전 조건은 다르다

Memory poisoning을 구체적인 공격 기법으로 제시한 연구는 <a href="https://arxiv.org/abs/2503.03704" target="_blank" rel="noopener">MINJA(Memory INJection Attack)</a>입니다. 이 논문의 고유한 기여는 수치보다 위협 모델 쪽에 있습니다. 공격자가 메모리 뱅크를 읽거나 직접 수정할 수 없고, 시스템 프롬프트를 건드릴 수도 없고, 사용자를 사칭할 수도 없는 상태에서, 오직 자기 계정으로 정상적인 쿼리를 보내고 응답을 관찰하는 것만으로 다른 사용자에게 영향을 주는 메모리 레코드를 심을 수 있다는 것입니다. 백엔드 권한이나 별도 인증이 필요 없기 때문에, 메모리가 사용자 간에 공유되거나 교차 참조되는 구조이고 공격자가 오직 쿼리만으로 상호작용하는 이 위협 모델 조건에서는, 그 에이전트를 정상적으로 쓸 수 있는 사람이면 누구나 공격자가 될 수 있습니다.

방법 쪽에서 논문이 쓰는 장치는 세 개입니다. 피해자의 쿼리와 악성 추론 단계를 연결하는 bridging step을 만들고, 에이전트가 그 bridging step을 스스로 생성하도록 유도하는 indication prompt를 붙이고, 그다음 indication prompt를 점진적으로 제거해(progressive shortening) 남은 레코드가 무관한 쿼리에도 자연스럽게 검색되도록 만듭니다. 눈에 띄는 흔적을 남기지 않고 검색 가능한 상태로 정착시키는 절차라고 볼 수 있습니다.

논문이 보고한 수치는 세 종류의 에이전트(EHRAgent, RAP, QA Agent)와 네 개 데이터셋을 GPT-4 계열 모델로 평가한 결과를 전체 평균한 것으로, injection 성공률(ISR) 98.2퍼센트, 공격 성공률(ASR) 76.8퍼센트입니다. 개별 조건의 분산은 작지 않습니다. 같은 표에서 EHRAgent와 MIMIC-III 조합의 ASR은 57.0퍼센트, RAP과 Webshop 조합은 GPT-4o에서 98.9퍼센트로 나타납니다. 즉 98.2퍼센트라는 숫자는 특정 시스템에서 기대할 값이 아니라 여러 조건을 평균한 값입니다.

조건에 따른 차이는 이후 연구에서 더 분명해집니다. <a href="https://arxiv.org/abs/2601.05504" target="_blank" rel="noopener">후속 EHR 실험 연구</a>는 MINJA 방식의 공격을 전자건강기록(EHR) 에이전트와 MIMIC-III 데이터셋 위에서 다시 실험하면서, 초기 메모리 상태와 indication prompt 개수, 최대 검색 관련 메모리 개수를 바꿔봤습니다. 이 연구는 MINJA가 보고한 성과를 "이상적인 조건에서 injection 성공률 95퍼센트 초과, 공격 성공률 70퍼센트"로 요약한 뒤, 조건을 현실 쪽으로 옮기면 결과가 달라진다는 것을 보입니다. GPT-4o-mini에 초기 메모리 6건, indication prompt 2개(progressive shortening 적용), 최대 검색 관련 메모리 개수 3을 둔 조건, 즉 이미 정상적인 기록이 존재하는 조건에서 ISR은 26.67퍼센트, ASR은 6.67퍼센트로 측정됐습니다. 같은 초기 메모리 6건에 indication prompt를 4개로 늘리고 최대 검색 관련 메모리 개수를 10으로 늘린 조건에서는 ISR이 100퍼센트, ASR이 38퍼센트로 다시 올라갑니다. 이 연구는 indication prompt를 2개에서 4개로 늘리는 변화 자체에는 유의미한 차이가 없었다고 밝히므로, 두 결과의 차이를 만드는 실질적인 변수는 최대 검색 관련 메모리 개수입니다. 즉 이 수치는 "MINJA가 실제로는 약하다"가 아니라 "검색 설정과 초기 메모리에 크게 의존한다"로 읽어야 합니다.

이 결과에서 얻을 수 있는 방어 방향은 정상 기록의 존재가 일종의 완충 역할을 한다는 점입니다. 그렇다면 방어 설계도 그 완충을 인위적으로 두껍게 하는 쪽, 즉 신뢰할 수 있는 기록에 가중치를 주고 의심스러운 기록을 검색 후보에서 밀어내는 쪽으로 갈 수 있습니다. 같은 연구가 설계하고 평가한 방어책 두 가지가 이 방향에 있습니다.

## 상용 제품에서 시연된 형태: SpAIware

논문 밖의 사례로 자주 인용되는 것이 보안 연구자 Johann Rehberger가 2024년 9월 공개한 <a href="https://embracethered.com/blog/posts/2024/chatgpt-macos-app-persistent-data-exfiltration/" target="_blank" rel="noopener">SpAIware</a>입니다. 실제 사용자가 피해를 입은 침해 사고가 보고된 것은 아니고, 연구자가 통제된 환경에서 구성해 책임 있는 공개 절차를 거쳐 발표한 proof-of-concept 시연입니다. 상용 제품에서 이 위협이 성립한다는 것을 보였다는 점에서 논문의 실험과는 다른 무게를 갖습니다.

공격 경로는 신뢰되지 않은 웹 페이지나 문서에 숨겨진 지시 사항이 ChatGPT의 메모리 도구를 호출하도록 유도하는 것으로 시작합니다. 이 지시가 성공하면 공격자가 원하는 내용이 사용자의 장기 메모리에 저장되고, 그 이후로는 사용자가 그 페이지를 다시 열지 않아도 저장된 지시가 대화마다 다시 작동합니다. Rehberger가 시연한 형태에서는 보이지 않는 이미지 렌더링을 이용해 대화 내용을 공격자 서버로 보내는 유출 경로가 여기에 결합되어 있었습니다.

Rehberger는 ChatGPT 버전 1.2024.247에서 이 취약점이 완화됐다고 기록하면서, 완화된 것이 유출 경로에 한정된다는 점을 함께 밝힙니다. 이미지 렌더링을 통해 제3자 서버로 데이터를 보내는 부분은 막혔지만, 원문의 표현으로는 신뢰되지 않은 웹 페이지나 문서가 여전히 메모리 도구를 호출해 임의의 내용을 저장할 수 있는 상태였습니다. 유출 채널과 쓰기 채널이 별개라는 뜻이고, 이 구분은 방어 설계에 그대로 옮겨집니다. 눈에 보이는 피해 경로 하나를 닫는 것과, 애초에 오염된 쓰기가 성립하지 않게 만드는 것은 다른 작업입니다.

## OWASP 분류에서 memory poisoning의 위치

이 위협은 업계 표준 분류에도 자리를 얻었는데, 한 항목에만 대응되지는 않습니다. 두 개의 OWASP 목록이 서로 연결된 상태로 각각 다른 범위를 담당합니다.

<a href="https://genai.owasp.org/llm-top-10/" target="_blank" rel="noopener">OWASP Top 10 for LLM Applications</a>에는 <strong>LLM04:2025 Data and Model Poisoning</strong>이 있습니다. 학습·파인튜닝·임베딩 데이터가 오염되는 문제를 다루는 항목이라 데이터 오염이라는 큰 틀은 겹치지만, 런타임에 에이전트의 지속 메모리에 기록이 추가되는 상황을 정면으로 겨냥한 항목은 아닙니다.

그쪽을 담당하는 것이 2025년 12월에 발표된 <a href="https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/" target="_blank" rel="noopener">OWASP Top 10 for Agentic Applications</a>의 <strong>ASI06:2026 Memory & Context Poisoning</strong>입니다. 대화 이력, RAG 인덱스, 임베딩, 지속 컨텍스트 저장소가 오염되어 이후의 추론과 계획, 도구 호출이 왜곡되는 상황을 다룹니다. 두 목록은 단절된 것이 아니라 상호 참조 관계로 설계되어 있습니다. <a href="https://cornucopia.owasp.org/cards/AAI3" target="_blank" rel="noopener">OWASP Cornucopia의 Agentic AI 카드 AAI3</a>는 같은 메모리 오염 시나리오를 LLM04:2025와 ASI06:2026 양쪽에 동시에 매핑합니다.

정리하면 memory poisoning은 기존 LLM 위험 분류에서 완전히 새로운 것이 아니라, 프롬프트 하나를 처리하는 시점 중심으로 짜인 분류에서는 다루기 어려운 지속성 문제 때문에 에이전트 전용 항목이 따로 필요해진 경우로 보는 편이 정확합니다.

## 설계·운영 원칙: 저장 전 검증에서 감사까지

지금까지의 근거를 모으면 방어의 큰 방향이 나옵니다. 메모리 쓰기와 읽기를 각각 별도의 검증 지점으로 취급하는 것입니다. 다만 어디까지가 논문이 실제로 구현하고 측정한 것이고 어디부터가 일반적인 엔지니어링 권고인지는 구분해 두는 편이 좋습니다.

### 논문이 직접 평가한 방어

앞서 살펴본 후속 EHR 연구가 설계하고 실험한 방어는 두 갈래입니다.

<strong>입출력 모더레이션과 복합 신뢰 점수.</strong> 정적 휴리스틱과 키워드 매칭, LLM 기반 의미 분류를 결합한 두 단계 게이트입니다. 입력 단계에서는 의심스러운 지시 문구나 제어 문자를 걸러 위험도가 높은 입력은 모델 호출 자체를 건너뛰고, 출력 단계에서는 생성된 코드에 위험한 연산이 있는지 검사해 각 항목을 감점 요소로 합산합니다. 이렇게 만든 복합 신뢰 점수가 threshold를 넘을 때만 메모리 추가를 허용합니다. 여러 독립적인 신호를 합산하는 방식이 단일 규칙 하나보다 넓은 범위를 잡아낼 수 있지만, 이것이 우회 난도를 얼마나 높이는지는 이 연구의 실험 범위에서 확정적으로 말하기 어렵습니다.

<strong>Trust-aware retrieval을 결합한 메모리 sanitization.</strong> 저장 시점에는 코드 안전성, 의미적 관련성, 정답 일치 여부를 조합해 0에서 1 사이의 연속 신뢰 점수를 매기고, 알려진 오염 템플릿에 해당하는 패턴을 정적으로 걸러냅니다. 검색 시점에는 오래된 항목의 신뢰 점수를 깎는 temporal decay를 적용하고, threshold 미달 항목을 few-shot 후보에서 제외하며, 유사도와 신뢰 점수를 함께 써서 순위를 정합니다.

이 연구가 강조하는 지점은 threshold 캘리브레이션이 까다롭다는 것입니다. 실제 평가에서 GPT-4o-mini는 후보 항목 23건을 전부 거부해 지나치게 보수적인 쪽으로 기울었고, Gemini-2.0-Flash는 신뢰도 1.0으로 82건을 통과시켰는데 그중 54건이 나중에 악성으로 판정됐습니다. 저자들은 후자를 보안 필터가 아니라 확신도 필터가 작동한 실패 양상으로 설명합니다. 즉 두 방어책 모두 threshold를 어떻게 잡느냐에 성패가 걸려 있고, 이 연구는 그 지점을 향후 적응형 방어를 위한 기준점으로 남겨 둡니다.

### 이를 확장한 운영 설계 예시

아래 세 가지는 위 논문들이 평가한 방어가 아니라, 같은 문제를 프로덕션에서 다룰 때 일반적으로 따라오는 엔지니어링 권고입니다. 효과를 측정한 근거는 아직 없으니 설계 시 검토할 항목 정도로 읽으시면 좋겠습니다.

<strong>출처·권한 분리(provenance separation).</strong> Untrusted input과 trusted memory를 같은 권한 층위에서 다루지 않는 접근입니다. 웹 페이지에서 읽은 내용과 사용자가 직접 확인한 내용에 저장 시점부터 다른 신뢰 등급을 부여하고, 낮은 등급의 항목이 높은 등급의 판단에 그대로 반영되지 않도록 경로를 나눠 두는 방식입니다. Dash et al.이 지적한 "쓰기 경로에 검증 단계가 없다"는 구조적 문제에 대응하는 방향이기는 하지만, 구체적인 등급 체계나 그 효과는 논문에서 평가되지 않았습니다.

<strong>TTL.</strong> 모든 메모리 항목에 유효기간을 두어, 오래전에 주입된 항목이 아무 검증 없이 영구히 남는 상황을 줄이는 방법입니다. 위 논문의 temporal decay가 신뢰 점수를 서서히 깎는 것이라면, TTL은 항목 자체를 만료시키는 더 강한 조치에 해당합니다. 도메인에 따라 오래된 정상 정보까지 함께 잃을 수 있어 트레이드오프를 따져봐야 합니다.

<strong>전체 감사 로그.</strong> 모든 write와 retrieval, 신뢰 점수 판정을 남겨 오염 경로를 사후에 추적할 수 있게 하는 것입니다. 위 연구도 모더레이션 판정을 감사 가능한 형태로 기록한다고 언급하지만, 파이프라인 전체를 로그로 덮는 설계는 별개의 운영 요구사항입니다. 로그가 없으면 threshold를 어느 방향으로 옮겨야 할지 판단할 재료 자체가 없습니다.

![저장 전 trust scoring, 출처·권한 분리, TTL과 temporal decay를 적용한 메모리 저장소, trust-aware retrieval, 감사 로그로 이어지는 방어 파이프라인 다이어그램](/ai-tech-blog/images/agent-memory-poisoning/diagram-2-defense-pipeline.png)

*방어 파이프라인. 논문이 평가한 요소(trust scoring, temporal decay, 패턴 필터링, trust-aware retrieval)와 운영 확장 요소(출처·권한 분리, TTL, 전체 감사 로그)를 한 흐름에 배치했습니다.*

이 항목들을 관통하는 변수는 결국 threshold 캘리브레이션입니다. 기준을 너무 보수적으로 잡으면 정상적인 메모리 갱신까지 막히고, 너무 느슨하게 잡으면 방어가 이름만 남습니다. 후속 EHR 연구에서 초기 메모리 구성과 검색 항목 수에 따라 공격 성공률이 크게 흔들렸다는 점을 함께 놓고 보면, threshold를 고정값으로 두기보다 축적된 정상 기록의 상태와 검색 설정을 함께 고려해 조정하는 편이 합리적일 것입니다. 다만 이 조정 방식 자체는 아직 검증된 레시피가 아닙니다.

## 엔지니어 관점 체크리스트

- 메모리 쓰기 채널을 명시적 지시, 시스템 프롬프트 유도, 압축 요약, 경험 일반화 네 가지로 나눠 각각 별도의 검증 로직이 있는지 확인합니다.
- Untrusted input에서 나온 정보와 사용자가 직접 확인한 정보가 저장 시점부터 다른 신뢰 등급을 갖는지 확인합니다.
- 모든 메모리 항목에 만료 시점이나 decay 곡선이 설정되어 있는지 확인합니다.
- 검색 단계에서 신뢰 점수가 낮은 항목을 배제하는 로직이 저장 단계의 필터와 별도로 존재하는지 확인합니다.
- Write와 retrieval, threshold 판정이 모두 감사 로그로 남아 사후 추적이 가능한지 확인합니다.
- 기존 prompt injection 방어 도구를 memory poisoning 방어로 그대로 재사용하고 있지 않은지 점검합니다.

## 한계와 열린 질문

이 글에서 다룬 방어 원칙들은 논문 수준에서 제안된 것이지, 대규모 프로덕션 환경에서 장기간 검증된 것은 아닙니다. Threshold를 얼마로 잡아야 하는지에 대한 일반적인 답은 아직 없고, 도메인과 사용 패턴에 따라 다시 튜닝해야 할 가능성이 큽니다. 또한 신뢰 점수를 산출하는 로직 자체가 별도의 공격 표면이 될 수 있다는 점도 남는 문제입니다. 공격자가 신뢰 점수 산출 방식을 역으로 추정해 점수를 높게 받는 방식으로 injection을 설계한다면, 방어 로직이 다시 공격 대상이 됩니다.

지속 메모리가 에이전트를 유용하게 만드는 이유와 공격 표면이 되는 이유는 같습니다. 세션 경계를 넘어 정보를 이어 준다는 점입니다. 그래서 방어를 사고 이후에 붙이는 필터로 처리하기가 특히 어렵습니다. 쓰기와 읽기를 설계 초기부터 각각의 검증 지점으로 잡아 두는 편이, 이미 쌓인 메모리에서 오염 항목을 뒤늦게 골라내는 작업보다 비용이 적게 듭니다.

## References

- Dash, P., Ge, T., Jain, A., Shah, T., & Shang, Z. (2026). *From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents.* Huawei Canada / University of Waterloo. arXiv:2606.04329. [arxiv.org/abs/2606.04329](https://arxiv.org/abs/2606.04329)
- Dong, S., Xu, S., He, P., Li, Y., Tang, J., Liu, T., Liu, H., & Xiang, Z. J. (2025). *Memory Injection Attacks on LLM Agents via Query-Only Interaction.* Advances in Neural Information Processing Systems 38, NeurIPS 2025 Main Conference Track. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42a97bbd9844d2bf68596730af80bcdf-Abstract-Conference.html) / arXiv:2503.03704. [arxiv.org/abs/2503.03704](https://arxiv.org/abs/2503.03704)
- Devarangadi Sunil, B., Sinha, I., Maheshwari, P., Todmal, S., Mallik, S., & Mishra, S. (2026). *Memory Poisoning Attack and Defense on Memory Based LLM-Agents.* arXiv:2601.05504. [arxiv.org/abs/2601.05504](https://arxiv.org/abs/2601.05504)
- OWASP GenAI Security Project. (2025년 12월 발표, ASI06:2026). *OWASP Top 10 for Agentic Applications, ASI06: Memory & Context Poisoning.* [genai.owasp.org](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)
- Habler, I. (2026). *Memory Is a Feature. It Is Also an Attack Surface.* OWASP GenAI Security Project Blog. [genai.owasp.org](https://genai.owasp.org/2026/05/13/memory-is-a-feature-it-is-also-an-attack-surface/)
- OWASP. *OWASP Top 10 for LLM Applications 2025, LLM04:2025 Data and Model Poisoning.* [genai.owasp.org/llm-top-10](https://genai.owasp.org/llm-top-10/)
- OWASP Cornucopia. *Agentic AI card AAI3.* [cornucopia.owasp.org](https://cornucopia.owasp.org/cards/AAI3)
- Rehberger, J. (2024). *Spyware Injection Into Your ChatGPT's Long-Term Memory (SpAIware).* Embrace The Red. [embracethered.com](https://embracethered.com/blog/posts/2024/chatgpt-macos-app-persistent-data-exfiltration/)

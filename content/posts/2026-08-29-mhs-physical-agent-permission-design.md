---
title: "AI 에이전트가 실제 기계를 움직이기 시작했다: Model Hardware Standard와 물리적 권한 설계"
date: 2026-08-29T09:00:00+09:00
draft: false
categories: ["Physical AI", "AI 에이전트"]
tags: ["Model Hardware Standard", "MHS", "Physical AI", "Agentic AI", "로봇 안전", "Functional Safety", "MCP", "제조 AI"]
author: "Jesam Kim"
slug: "mhs-physical-agent-permission-design"
cover:
  image: "/ai-tech-blog/images/mhs-physical-agent-permission-design/cover.png"
  alt: "빛나는 경계선 앞에 멈춘 로봇팔이 놓인 실험실"
  relative: false
description: "Anthropic이 2026년 8월 27일 공개한 Model Hardware Standard research preview를 근거로, 에이전트가 물리 장치를 조작할 때 권한 경계가 어디서 달라지는지 정리합니다. 공식 발표에서 확인되는 사실과 제조 및 엔터프라이즈 관점의 설계 원칙을 구분해서 다룹니다."
---

소프트웨어 에이전트의 tool call이 실패했을 때 우리가 실제로 하는 일은 재시도입니다. API가 500을 반환하면 다시 호출하고, 파일 쓰기가 깨지면 이전 커밋으로 되돌리고, 잘못된 레코드가 들어가면 트랜잭션을 롤백합니다. 이 습관은 대상이 비트라는 사실에 기대고 있습니다. 비트는 복제할 수 있고, 이전 상태를 보관할 수 있고, 같은 명령을 두 번 실행해도 대체로 같은 결과에 도달합니다.

로봇팔을 5cm 움직이는 명령은 이 전제 중 어느 것도 만족하지 않습니다. 명령이 절반만 실행된 상태에서 재시도하면 팔은 10cm를 움직입니다. 시약 40µL를 분주한 뒤에는 되돌릴 이전 상태가 없고, 시료 자체가 소모됩니다. 레이저 출력을 잘못 올리면 형광 분자가 표백되어 그날의 실험이 끝납니다. 소프트웨어 에이전트 설계에서 익숙해진 "일단 시도하고 실패하면 되돌린다"는 패턴이, 물리 장치에서는 그 자체로 사고의 경로가 됩니다.

[Anthropic이 2026년 8월 27일 research preview로 공개한 Model Hardware Standard(MHS)](https://www.anthropic.com/news/model-hardware-standard-research-preview)는 이 경계를 실무 문제로 끌어왔습니다. 이 글에서는 발표문에서 확인할 수 있는 사실을 먼저 정리하고, 그다음에 제조와 엔터프라이즈 환경에서 권한을 어떻게 끊어야 하는지를 저자의 관점으로 이어갑니다. 두 부분은 근거의 성격이 다르므로 절을 나누어 표시했습니다. 확인되지 않은 사양은 이 글에서 다루지 않습니다.

## MHS가 실제로 공개한 범위

MHS는 표준화된 드라이버입니다. 발표문의 설명으로는 컴퓨터의 운영체제와 하드웨어 장치 사이를 번역하는 소프트웨어이고, `read`(예를 들어 "온도를 읽어라")와 `write`(예를 들어 "온도를 설정하라") 같은 단순한 primitive를 사용합니다. 각 장치는 표준 형식으로 discoverable해지므로, 장치와 에이전트가 중간 번역 프로그램 없이 네트워크를 넘어 서로를 찾습니다.

드라이버가 하는 일이 하나 더 있습니다. 코드만 봐서는 알 수 없는 장치 특성을 자연어 태그로 적어 넣게 하고, 그 태그에서 참조 파일을 자동으로 만듭니다. 발표문이 든 예시는 로봇팔의 무게입니다. 팔을 안전하게 조작하려면 알아야 하지만 API 시그니처에는 없는 정보이고, 지금까지는 종이 매뉴얼이나 담당자의 머릿속에 있었습니다. 생성되는 참조 파일에는 장치가 무엇을 측정할 수 있는지, 무엇을 조정할 수 있는지, 그리고 어떤 안전 한계가 강제되는지가 담깁니다.

제어 경로는 세 가지입니다. [MCP](https://modelcontextprotocol.io), 커맨드라인 인터페이스, 그리고 코드 파일 형태의 API입니다. MHS는 model-agnostic이고, MCP 같은 표준 프로토콜을 쓰는 어떤 agent harness에서도 접근할 수 있다고 명시되어 있습니다. 에이전트가 매 단계를 추론하기에는 너무 길거나 빠른 작업은 드라이버 명령을 코드 파일로 묶어서 장치가 스스로 수행하게 합니다.

출발점은 표준 제정 프로젝트가 아니었습니다. HHMI Janelia Research Campus의 박사후연구원 Arco Bast가 벤더가 다른 레이저, 모터 포커서, 카메라를 같은 rig에서 돌리려고 공유 메모리 딕셔너리를 만든 것이 시작이고, Anthropic Beneficial Deployments 팀의 Alek Kemeny가 여기에 모델을 결합했습니다. Bast의 커스텀 현미경이 MHS로 동작한 첫 rig입니다.

발표문에 함께 실린 파트너 사례에서 권한과 검증에 직접 걸리는 대목을 뽑으면 다음과 같습니다.

| 파트너 | 확인되는 내용 |
|---|---|
| Carnegie Mellon University | 세 대의 컴퓨터에 흩어진 액체 핸들러, 플레이트 리더, 로봇팔, 카메라를 통합하고 첫 희석 곡선을 얻는 데까지 약 8시간이 걸렸습니다. 벤더에 맡기면 보통 몇 주가 걸리는 작업입니다. plate 없음, plate 회전, 리더 사용 중, 카메라 연결 끊김, 장치 도달 불가, emergency stop 활성의 여섯 조건을 인위적으로 만들었을 때 <strong>여섯 건 모두 장치가 움직이기 전에 차단</strong>되었습니다. |
| QuEra Computing | 사람이 만든 relock 스크립트는 성공률 약 58%에 시도당 약 150초였습니다. 에이전트가 밤새 반복 개선한 스크립트는 블라인드 테스트 700회 중 695회 성공(99.3%)했고, 최종 산출물은 에이전트 없이 돌아가는 결정론적 스크립트입니다. |
| HHMI Janelia | MHS가 장치 단위 안전 한계를 강제하므로, 에이전트가 레이저 출력을 과도하게 쓸 걱정을 하지 않았다고 연구자가 서술합니다. |
| Genentech | 거품 때문에 발생한 런타임 오류에서 Claude의 기본 반응은 같은 well에서 파라미터만 바꿔 재시도하는 것이었고, 그 결과 거품이 더 생겼습니다. 물리적 실패라는 사실을 사람이 알려준 뒤에야 교정되었습니다. |

여기에 하나를 덧붙일 필요가 있습니다. QuEra는 [자사 블로그](http://www.quera.com/blog-posts/holding-the-light-teaching-an-ai-to-lock-and-tune-our-quantum-computers-lasers)에서 MHS가 장치가 선언한 한계, 인터록, emergency stop을 하드웨어 인터페이스에서 모델과 독립적으로 강제한다고 서술합니다. 다만 이 서술은 파일럿을 수행한 파트너가 남긴 기록이므로, 사양의 규범적 요구사항인지 구현상의 성질인지는 공개 자료만으로 구분되지 않습니다.

접근 조건도 사실로 확인됩니다. MHS는 과학 연구실과 첨단 제조 업체를 대상으로 한 제한된 research preview이고, [공식 사이트](https://www.modelhardwarestandard.com/)에는 사양 문서 없이 신청 양식과 발표문 링크만 있습니다. Anthropic은 안전 평가와 모범 사례를 파트너와 함께 만든 뒤에 오픈소스로 공개하겠다고 밝혔고, 시점은 제시하지 않았습니다. 프로그래밍 인터페이스가 없는 장치는 아직 지원하지 않습니다.

하드웨어 벤더 쪽에서는 Universal Robots, Doosan Robotics, Tecan, QIAGEN, MBF Bioscience, Automata, Danaher가 지원이나 검토 단계로 이름을 올렸고, 소프트웨어 쪽에서는 Hugging Face의 LeRobot, Raspberry Pi, 그리고 AWS의 Strands Robots 라이브러리가 언급되었습니다. AWS는 preview 기간 동안 참가자에게 비공개 사전 배포 패키지를 제공한다고 적혀 있습니다.

![벤더별 glue code 구조와 MHS 드라이버 구조를 위아래로 비교했습니다. 위쪽은 에이전트가 장치마다 별도 glue code와 벤더 인터페이스를 거쳐 연결되고, 아래쪽은 MHS 표준 드라이버 한 겹과 하드웨어 인터페이스의 강제 경계를 거쳐 연결됩니다.](/ai-tech-blog/images/mhs-physical-agent-permission-design/mhs-driver-vs-glue-code.png)

<em>MHS 이전에는 통합 코드가 장치 수만큼 늘어나고 안전 한계도 그 코드에 흩어집니다. MHS에서는 드라이버 한 겹으로 모이고, 강제 지점이 모델 바깥에 놓입니다. 하드웨어 인터페이스가 인터록과 emergency stop까지 모델과 독립적으로 강제한다는 서술은 QuEra가 남긴 기록입니다.</em>

## read와 write 사이에 빠져 있는 구분

여기서부터는 저자의 분석입니다.

`read`와 `write`는 데이터가 흐르는 방향을 나타내는 구분이고, 위험 등급을 나타내는 구분이 아닙니다. 온도 설정값을 25도에서 26도로 바꾸는 것도 `write`이고, 6축 로봇팔을 사람이 서 있을 수 있는 좌표로 이동시키는 것도 `write`입니다. 두 명령은 프로토콜 관점에서 같은 종류이지만, 잘못됐을 때 남는 결과가 전혀 다릅니다. 권한 모델을 primitive 위에 그대로 얹으면 이 차이가 사라집니다.

물리 장치를 다루는 에이전트에서는 최소 세 가지로 갈라서 봐야 합니다.

<strong>관측(read)</strong>은 대체로 되돌릴 수 있지만 무료는 아닙니다. 형광 이미징에서 레이저를 한 번 더 쏘는 것은 데이터를 얻는 동작이면서 동시에 시료를 소모하는 동작입니다. Janelia 사례에서 안전 한계가 걸린 대상이 바로 레이저 출력이었다는 점이 이 성질을 보여줍니다. 관측 예산을 정해 두지 않으면, 더 확실한 판단을 위해 더 많이 관측하는 에이전트의 성향이 시료를 태웁니다.

<strong>구동(actuate)</strong>은 상태를 바꾸고, 대부분 되돌릴 수 없습니다. 분주된 시약, 소모된 시약 팁, 이미 회전한 원심분리기는 이전 상태로 복원되지 않습니다. 롤백할 대상이 없으므로 실행 전 검증이 그 자리를 대신합니다. CMU가 카메라로 plate의 존재와 방향을 확인한 뒤에야 이송을 허용한 것이 이 사전 조건 검사에 해당합니다.

<strong>안전 관련 동작(safety-critical)</strong>은 사람이나 설비에 위험을 만들 수 있는 동작입니다. 이 등급에서는 모델의 판단을 신뢰 경로에 넣지 않는 것이 원칙입니다. 판단이 틀릴 확률이 높기 때문이 아니라, 확률을 검증할 방법이 없기 때문입니다. 기능 안전 표준은 고장 모드와 진단 범위를 문서로 증명할 수 있는 구조를 요구하고, "대체로 옳은 판단"은 그 요구를 충족하지 못합니다.

MHS의 참조 파일은 장치가 무엇을 할 수 있고 어떤 안전 한계가 걸려 있는지를 기술합니다. 이것은 장치의 능력 기술이고, 특정 에이전트가 특정 시점에 그중 무엇을 해도 되는지를 정하는 권한 기술은 아닙니다. 두 문서를 따로 만들어야 합니다.

## 제조와 엔터프라이즈 환경의 권한 설계 원칙

### capability를 화이트리스트로 좁힌다

장치가 노출하는 procedure 전부를 에이전트에게 주지 않고, 워크플로에 필요한 것만 명시적으로 고릅니다. 기본값은 거부이고, 허용은 열거로만 이루어집니다. MHS가 장치 발견을 쉽게 만들었다는 점이 여기서 양날이 됩니다. 네트워크에 붙은 장치를 에이전트가 스스로 찾을 수 있다는 것은, 화이트리스트가 없으면 의도하지 않은 장치까지 조작 대상이 된다는 뜻입니다. Tetsuwan Scientific 사례에서 ResearchOS가 거품 문제를 해결할 장치를 찾으려고 네트워크를 스캔해 원심분리기를 찾아낸 것은 발견 기능의 유용함을 보여주는 동시에, 그 스캔 범위를 누가 정하는지가 설계 항목이 된다는 것도 보여줍니다.

### human approval 지점을 모델 재량에서 떼어낸다

승인을 언제 받을지 모델이 판단하게 하면 두 방향으로 다 실패합니다. QuEra 파일럿에서는 Claude가 조금이라도 위험해 보이는 동작 앞에서 자주 사람의 확인을 기다렸고, 그 때문에 실험이 밤새 멈춰 있는 일이 있었습니다. QuEra 팀은 그래도 지나치게 조심스러운 에이전트가 덜 조심스러운 에이전트보다 낫다고 정리했는데, 운영 관점에서는 무인 야간 운전의 근거가 사라지는 문제이기도 합니다. 반대로 모델이 스스로 안전하다고 판단해 승인을 건너뛰면 게이트가 없는 것과 같습니다.

승인 대상은 장치 쪽 정의로 고정하는 편이 낫습니다. procedure 단위로 승인 필요 여부를 선언하고, 에이전트는 그 선언을 바꿀 수 없게 합니다. CMU 팀도 향후 과제로 고위험 결정에서 사람의 승인이 언제 어떻게 필요한지에 대한 프로토콜을 정교하게 만들겠다고 적었습니다. 이것이 아직 정해지지 않은 항목이라는 뜻입니다.

### 시뮬레이션과 디지털 트윈을 실행 전 단계로 끼운다

Anthropic 발표문에는 시뮬레이션이나 디지털 트윈에 관한 서술이 없습니다. 이 항목은 저자의 제안입니다.

에이전트가 코드 파일로 묶어 내보내는 명령 시퀀스는 사람이 검토하기 전에 실물에 닿습니다. 로봇팔 궤적이 포함되면 충돌 검사가 필요하고, 이것은 실행 후 로그로는 확인할 수 없습니다. 기존 로봇 도입 공정에는 이미 오프라인 프로그래밍과 셀 시뮬레이션 단계가 있으므로, 단계를 새로 만들기보다 에이전트가 생성한 시퀀스를 그 단계에 통과시키는 연결을 만드는 쪽이 현실적입니다. dry-run 모드를 장치 드라이버 수준에서 제공할 수 있다면 더 좋습니다. 같은 명령을 구동 없이 검증만 하는 경로가 있으면, 에이전트가 자기 스크립트를 실물에 걸기 전에 스스로 확인할 수 있습니다.

### emergency stop과 인터록은 모델 밖에 둔다

emergency stop은 소프트웨어 요청으로 대체할 수 없는 기능입니다. [ISO 13850](https://www.iso.org/standard/59970.html)은 사용하는 에너지 종류와 무관하게 emergency stop 기능의 기능 요구사항과 설계 원칙을 규정하는 표준입니다. 원문은 유료 열람이므로 여기서는 표준이 다루는 범위까지만 적습니다. 에이전트를 붙일 때 확인해야 할 것은 정지 경로가 에이전트 경로와 물리적으로 분리되어 있는지, 그리고 정지가 걸린 상태에서 에이전트의 구동 명령이 거부되는지입니다. CMU가 emergency stop 활성 조건을 시험 항목에 포함한 이유도 두 번째 확인에 있습니다.

한 가지 유의점이 있습니다. MHS의 안전 한계는 장치가 선언하는 값입니다. 선언이 곧 강제이므로, 드라이버 작성자가 한계를 잘못 적으면 강제도 잘못 걸립니다. 자연어 태그로 하드웨어 특성을 적어 넣는 방식은 통합 속도를 크게 줄이는 대신, 그 텍스트가 안전 근거가 된다는 결과를 함께 만듭니다. 태그 작성과 검토를 별도 승인 절차로 다루어야 하고, 태그 파일의 변경 이력은 코드와 같은 수준으로 관리해야 합니다.

### fail-safe 상태를 장치별로 정의한다

"위험하면 멈춘다"가 모든 공정에서 안전한 선택은 아닙니다. 가열 중인 반응을 중간에 멈추면 시료가 상하고, 회전 중인 원심분리기를 급정지시키면 장비가 손상됩니다. University of Washington 사례가 이 지점을 보여줍니다. 에이전트가 qPCR 증폭 곡선을 읽다가 적절한 시점에 연구자에게 중단 여부를 묻고, 중단 지시를 받으면 반응을 정지한 뒤 장비를 4도 hold로 넘깁니다. DNA가 상하지 않는 상태로 옮겨두는 전이입니다. 판단은 사람이 하고 전이는 정해진 프로토콜이 수행한다는 점이 중요합니다. 장치마다 이 상태를 미리 정의하고, 에이전트가 이상을 감지했을 때 진입할 경로를 결정론적 코드로 고정해 두어야 합니다.

### 궤적 단위로 감사 기록을 남긴다

소프트웨어 에이전트 쪽에서는 호출 한 건씩 검사하는 방식의 한계가 이미 논의되고 있습니다. 개별 호출이 모두 허용 범위여도 순서가 쌓이면 제약을 위반할 수 있다는 문제이고, 이 블로그에서도 [궤적 단위 검증](/ai-tech-blog/posts/trajectory-assurance-agent-security/)으로 한 번 다뤘습니다. 물리 장치에서는 여기에 조건이 하나 더 붙습니다. 명령 로그만으로는 사고를 재현할 수 없고, 그 시점의 센서값이 함께 있어야 합니다. 로봇팔이 왜 그 좌표로 갔는지는 명령에 남지만, 그때 작업 공간에 무엇이 있었는지는 센서에만 남습니다.

기록해야 할 항목은 명령, 명령을 낸 주체, 근거가 된 관측값, 실행 시점의 장치 상태, 그리고 결과입니다. QuEra 파일럿에서 에이전트 루프의 한 역할이 모든 단계를 기록하는 logbook이었고 다른 역할이 그 logbook을 읽어 다음 변경을 결정했다는 구조가, 감사 기록이 운영 입력으로도 쓰인다는 것을 보여줍니다. MHS 공개 자료에는 감사 로그의 표준 형식에 관한 서술이 없으므로, 지금은 각 조직이 스스로 정해야 하는 항목입니다.

![물리 에이전트의 권한 게이트를 세로 흐름으로 그렸습니다. 에이전트 의도, capability 화이트리스트, 위험 등급 분류, 등급별 게이트, 장치 선언 한계 강제, 실행, 물리 장치, 궤적 감사 기록으로 이어지고, 오른쪽에는 emergency stop과 인터록이 모델을 거치지 않고 장치로 직결되는 별도 경로가 있습니다.](/ai-tech-blog/images/mhs-physical-agent-permission-design/permission-gate-flow.png)

<em>왼쪽 흐름은 에이전트가 지나야 하는 게이트이고, 오른쪽 붉은 경로는 모델 판단을 거치지 않는 정지 경로입니다. 이 구성은 저자의 설계안이며, MHS 사양이 요구하는 구조가 아닙니다.</em>

## MHS에서 아직 확인할 수 없는 것

<strong>사양이 공개되어 있지 않습니다.</strong> 드라이버 스키마, 참조 파일 형식, discovery의 전송 방식과 범위 제한이 모두 신청자에게만 열려 있습니다. 표준의 설계를 평가하려면 문서가 필요한데, 지금 판단 근거는 발표문과 파트너 후기입니다.

<strong>인증과 권한 위임 모델이 제시되지 않았습니다.</strong> 장치와 에이전트가 네트워크를 넘어 서로를 찾는다면, 어떤 에이전트인지 확인하는 절차와 자격 증명을 회수하는 절차가 필요합니다. 공개 자료에는 이 부분에 관한 서술이 없습니다. 소프트웨어 프로토콜에서는 인증을 나중에 붙여도 자격 증명을 재발급하면 수습되지만, 물리 장치에서는 잘못 위임된 권한이 이미 움직인 기계로 남습니다.

<strong>기능 안전 표준과의 관계가 명시되지 않았습니다.</strong> [ISO 10218-1](https://www.iso.org/standard/73933.html)과 [ISO 10218-2](https://www.iso.org/standard/73934.html)는 2025년에 개정되었고, [EU 기계 규정 2023/1230](https://eur-lex.europa.eu/eli/reg/2023/1230/oj)은 2027년 1월 20일부터 적용됩니다. MHS의 "장치 선언 안전 한계"가 이 체계에서 어떤 지위인지, 인증을 받은 안전 기능을 대체할 수 있는지 아니면 그 위에 얹히는 것인지는 공개 자료로 알 수 없습니다. 현재로서는 후자로 보고 설계하는 것이 안전합니다.

<strong>물리 추론의 한계가 남아 있습니다.</strong> 이 제약은 모델에서 오고, Anthropic도 발표문에서 직접 인정합니다. Claude는 물리 세계를 텍스트와 이미지로 학습하므로 공간 추론과 물리 추론에 한계가 있고 전문가의 감독이 필요하다는 서술입니다. Genentech의 거품 사례와 QuEra의 "물리 하드웨어에 문제가 생기면 Claude가 원인을 찾지 못했다"는 기록이 같은 지점을 가리킵니다.

<strong>통합 시간 단축이 검토 부담으로 옮겨갑니다.</strong> 이 항목은 저자의 관측입니다. 몇 주에서 몇 시간으로 줄어든 것은 통합 작업이고, 위험 평가는 함께 줄지 않습니다. 오히려 하루에 장치 여섯 대를 붙일 수 있게 되면 위험 평가를 거치지 않은 장치가 쌓이는 속도가 빨라집니다. 통합이 병목이던 조직에서는 병목이 검토로 옮겨가고, 그 검토를 누가 언제 하는지를 정하지 않으면 검토가 생략됩니다.

## 표준보다 권한 경계와 검증 루프가 먼저다

MHS가 해결한 문제는 실재하고 컸습니다. 벤더가 다른 장치 여섯 대를 드라이버 작성 시간까지 포함해 일주일 안에 붙였다는 University of Washington의 기록, 프로그램 일곱 개를 정해진 순서로 띄우던 절차가 대시보드 클릭 한 번이 되었다는 Janelia의 기록은 인터페이스 표준화가 만드는 차이를 분명하게 보여줍니다. 통합이 어려워서 포기했던 실험들이 가능해집니다.

그런데 인터페이스가 통일되어도 권한은 통일되지 않습니다. 표준이 정하는 것은 명령을 어떻게 전달하느냐이고, 누가 어떤 명령을 언제 내려도 되느냐는 여전히 각 조직이 정해야 합니다. 물리 장치에서는 이 결정이 사후 정리 항목이 될 수 없습니다. 잘못된 write 한 번이 시료를 태우거나 설비를 부수거나 사람을 다치게 하고, 이 중 어느 것도 롤백되지 않습니다.

지금 MHS 도입을 검토하는 조직이 표준 문서를 기다리는 동안 할 수 있는 일은 정해져 있습니다. 장치별로 procedure를 관측, 구동, 안전 관련으로 분류한 표를 만들고, 각 등급에 승인 규칙을 붙이고, emergency stop 경로가 에이전트 경로와 분리되어 있는지 확인하고, 명령과 센서값을 함께 남기는 감사 스키마를 정하는 것입니다. 이 네 가지는 어떤 표준을 쓰든 필요하고, 표준이 확정되기를 기다릴 이유가 없습니다. Anthropic이 research preview 기간에 파트너와 함께 만들겠다고 한 것도 정확히 이 영역의 안전 평가와 모범 사례입니다.

## References

- [Previewing the Model Hardware Standard](https://www.anthropic.com/news/model-hardware-standard-research-preview), Anthropic, 2026-08-27
- [Model Hardware Standard 공식 사이트](https://www.modelhardwarestandard.com/), Anthropic
- [Anthropic's new hardware standard lets AI agents control the physical world](https://arstechnica.com/ai/2026/08/anthropics-new-hardware-standard-lets-ai-agents-control-the-physical-world/), Kyle Orland, Ars Technica, 2026-08-27
- [Anthropic pushes into physical world with new standard to help AI agents operate machines](https://www.cnbc.com/2026/08/27/anthropic-pushes-into-physical-world-with-new-standard-to-help-ai-agents-operate-machines.html), Ashley Capoot, CNBC, 2026-08-27
- [Holding the Light: Teaching an AI to Lock and Tune our Quantum Computer's Lasers](http://www.quera.com/blog-posts/holding-the-light-teaching-an-ai-to-lock-and-tune-our-quantum-computers-lasers), QuEra Computing, 2026-08-27
- [Model Hardware Standard 적용 사례](http://tetsuwan.com/blog/mhs), Tetsuwan Scientific
- [Model Context Protocol](https://modelcontextprotocol.io), MCP 공식 문서
- [HHMI Janelia Research Campus](https://www.hhmi.org/research/janelia)
- [ISO 10218-1:2025, Robotics, Safety requirements, Part 1: Industrial robots](https://www.iso.org/standard/73933.html), ISO
- [ISO 10218-2:2025, Robotics, Safety requirements, Part 2: Industrial robot applications and robot cells](https://www.iso.org/standard/73934.html), ISO
- [ISO 13850:2015, Safety of machinery, Emergency stop function, Principles for design](https://www.iso.org/standard/59970.html), ISO
- [Regulation (EU) 2023/1230 on machinery](https://eur-lex.europa.eu/eli/reg/2023/1230/oj), EUR-Lex
- [Building intelligent physical AI: From edge to cloud with Strands Agents, Bedrock AgentCore, Claude 4.5, NVIDIA GR00T, and Hugging Face LeRobot](https://aws.amazon.com/blogs/opensource/building-intelligent-physical-ai-from-edge-to-cloud-with-strands-agents-bedrock-agentcore-claude-4-5-nvidia-gr00t-and-hugging-face-lerobot/), AWS Open Source Blog

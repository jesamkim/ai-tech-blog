---
title: "AWS와 Physical AI: Amazon이 로봇과 물리 세계를 정복하는 법 — RoboMaker부터 100만 로봇 배치까지"
date: 2026-02-17T12:54:51+09:00
draft: false
author: "Jesam Kim"
description: "AWS의 로보틱스 클라우드 인프라부터 Amazon 물류 현장의 100만 로봇 운용까지, Physical AI 시대를 선도하는 Amazon의 전략을 분석합니다."
categories:
  - "Physical AI"
tags:
  - "AWS"
  - "RoboMaker"
  - "Physical AI"
  - "Amazon Robotics"
  - "Embodied AI"
  - "클라우드 로보틱스"
ShowToc: true
TocOpen: true
---

## Physical AI란 무엇인가 — 소프트웨어 AI에서 물리 세계로의 확장

ChatGPT가 텍스트를 생성하고, Midjourney가 이미지를 만들어도 AI는 여전히 스크린 안에 갇혀 있었습니다. Physical AI(Embodied AI)는 이 경계를 허뭅니다. AI가 물리 환경을 인식(Perceive)하고, 판단(Reason)하고, 실제로 조작(Manipulate)하는 기술 패러다임입니다.

### 소프트웨어 AI vs Physical AI

![다이어그램 1](/ai-tech-blog/images/posts/2026-02-17/aws와-physical-ai-amazon이-로봇과-물리-세계를-정복하는-법-robomaker부터-100만-/diagram-1.png)

핵심 차이는 피드백 루프(Feedback Loop)에 있습니다. 소프트웨어 AI는 틀려도 텍스트를 다시 생성하면 그만입니다. 반면 Physical AI가 로봇 팔의 각도를 1도만 잘못 계산하면 물건이 깨지고, 사람이 다칩니다. 현실은 Undo가 없습니다.

### 빅테크 경쟁 구도 속 Amazon의 포지션

| 기업 | 전략 | 강점 |
|------|------|------|
| NVIDIA | Omniverse + Isaac Sim으로 시뮬레이션 플랫폼 장악 | GPU 생태계, Digital Twin |
| Google DeepMind | RT-2, Gemini Robotics로 범용 로봇 모델 추구 | 연구 역량, Foundation Model |
| Tesla | Optimus 휴머노이드로 제조업 직접 투입 | 자체 공장 = 테스트베드 |
| Amazon | 실제 물류 현장 100만+ 로봇 운영 + AWS 클라우드 | 배포 규모(Deployment at Scale) |

NVIDIA가 시뮬레이션 도구를, Google이 가장 똑똑한 두뇌를 만든다면, Amazon은 실전에서 매일 수억 개 패키지를 옮기는 근육을 이미 갖고 있습니다. 이론이 아니라 프로덕션 환경에서 검증된 Physical AI. 이것이 Amazon의 결정적 차별점입니다.

### 왜 "지금" Physical AI인가

세 가지 기술 곡선이 동시에 변곡점(Inflection Point)을 맞이했습니다.

```python
# Physical AI 성숙도를 결정하는 세 축
physical_ai_readiness = {
    "foundation_models": {
        "상태": "GPT-4급 멀티모달 모델이 로봇 태스크 플래닝 가능",
        "변화": "자연어 명령 → 로봇 액션 시퀀스 자동 생성"
    },
    "simulation": {
        "상태": "Isaac Sim, AWS RoboMaker 등 고충실도 물리 시뮬레이션 상용화",
        "변화": "실제 로봇 없이 수백만 에피소드 학습 (Sim-to-Real Transfer)"
    },
    "hardware_cost": {
        "상태": "로봇 팔 가격 $50K → $5K, 센서 원가 1/10로 하락",
        "변화": "중소기업도 로봇 도입 가능한 경제성 확보"
    }
}

# 세 조건이 모두 충족될 때 Physical AI가 폭발적으로 성장
is_ready = all(v["상태"] != "" for v in physical_ai_readiness.values())
print(f"Physical AI 시대 도래: {is_ready}")  # True
```

Foundation Model이 로봇의 두뇌를, 시뮬레이션이 훈련장을, 저가 하드웨어가 신체를 제공합니다. 개인적으로 이 세 조건이 동시에 갖춰진 건 이번이 처음이라고 봅니다. AI가 마침내 스크린 밖으로 나올 준비가 된 셈입니다.

그리고 Amazon은 이 삼각형의 모든 꼭짓점에 이미 투자를 마친, 거의 유일한 기업입니다.

## AWS RoboMaker에서 IoT RoboRunner까지 — 클라우드 로보틱스 플랫폼의 진화

앞서 Physical AI를 소프트웨어의 경계를 넘어 물리 세계와 상호작용하는 AI라고 정의했습니다. 그렇다면 이 Physical AI를 대규모로 개발하고, 테스트하고, 배포하는 인프라는 어떤 모습일까요? AWS는 지난 수년간 이 질문에 대한 답을 플랫폼 단위로 구축해 왔습니다.

### AWS RoboMaker — 클라우드 네이티브 로봇 개발의 시작

AWS RoboMaker는 ROS(Robot Operating System) 기반 로봇 애플리케이션을 클라우드에서 개발·시뮬레이션·배포할 수 있는 완전관리형 서비스입니다.

- 대규모 병렬 시뮬레이션(Simulation): 물리 로봇 없이 Gazebo 기반 가상 환경에서 수백 개의 시나리오를 동시에 테스트할 수 있습니다. 실제로 써보면 물리 로봇 한 대 없이도 꽤 다양한 엣지 케이스를 잡아낼 수 있어서, 초기 개발 단계에서 시간을 크게 절약해 줍니다.
- 플릿 관리(Fleet Management): OTA(Over-The-Air) 업데이트로 수천 대 로봇에 새 코드를 일괄 배포합니다.
- CI/CD 파이프라인: 코드 커밋부터 시뮬레이션 테스트, 실제 로봇 배포까지 자동화할 수 있습니다.

```python
# RoboMaker 시뮬레이션 작업 생성 예시
import boto3

client = boto3.client('robomaker')

response = client.create_simulation_job(
    maxJobDurationInSeconds=3600,
    iamRole='arn:aws:iam::012345678901:role/RoboMakerRole',
    simulationApplications=[{
        'application': 'arn:aws:robomaker:us-east-1:012345678901:simulation-application/my-sim-app/1',
        'launchConfig': {
            'packageName': 'warehouse_sim',
            'launchFile': 'warehouse_world.launch',
            'environmentVariables': {
                'NUM_ROBOTS': '50',
                'SCENARIO': 'peak_hour_stress_test'
            }
        }
    }],
    robotApplications=[{
        'application': 'arn:aws:robomaker:us-east-1:012345678901:robot-application/my-robot-app/1',
        'launchConfig': {
            'packageName': 'navigation_bot',
            'launchFile': 'nav_stack.launch'
        }
    }]
)
```

### IoT RoboRunner — 이기종 로봇 함대의 통합 오케스트레이션

현실의 물류 창고에는 서로 다른 제조사의 AMR(Autonomous Mobile Robot), 로봇 팔, 컨베이어 시스템이 뒤섞여 있습니다. IoT RoboRunner는 이 이기종(heterogeneous) 로봇 함대를 단일 인터페이스로 관리하는 오케스트레이션 레이어입니다. 각 로봇 벤더의 독자 API를 추상화해서 작업 할당(Task Allocation), 경로 최적화, 교착 상태(Deadlock) 회피를 중앙에서 제어합니다. 개인적으로는 벤더 락인 없이 여러 로봇을 하나의 창구로 다룰 수 있다는 점이 가장 실용적인 부분이라고 생각합니다.

### 풀스택 파이프라인 — 학습에서 엣지 배포까지

AWS 로보틱스 전략에서 눈여겨볼 부분은 개별 서비스보다 엔드투엔드 파이프라인입니다.

![다이어그램 2](/ai-tech-blog/images/posts/2026-02-17/aws와-physical-ai-amazon이-로봇과-물리-세계를-정복하는-법-robomaker부터-100만-/diagram-2.png)

SageMaker에서 강화학습(Reinforcement Learning)으로 로봇 제어 정책을 훈련합니다. 이후 RoboMaker에서 수백 가지 시뮬레이션 시나리오로 안전성을 검증하고, IoT Greengrass를 통해 엣지 디바이스인 실제 로봇에 모델을 배포합니다. 현장에서 수집된 운영 데이터는 다시 SageMaker로 피드백되어 모델을 개선하는 순환 구조를 이룹니다. 각 서비스가 따로 노는 게 아니라 하나의 루프로 연결된다는 점이, 실제 운영 환경에서 꽤 큰 차이를 만듭니다.

## Amazon Robotics — 100만 로봇이 움직이는 물류 현장의 실체

2012년 Amazon이 Kiva Systems를 7.75억 달러에 인수한 건 단순한 물류 투자가 아니었습니다. Physical AI를 대규모로 실증할 수 있는 실험장을 통째로 확보한 사건이었습니다.

### 로봇 라인업의 진화

지난 10년간 Amazon Robotics는 물류 공정의 병목마다 특화된 로봇을 하나씩 투입해 왔습니다.

| 로봇 | 도입 시기 | 역할 | 핵심 기술 |
|------|-----------|------|-----------|
| Kiva (현 Hercules) | 2012~ | 선반 운반 (Goods-to-Person) | 2D 바코드 내비게이션 |
| Robin | 2020~ | 패키지 분류 | Computer Vision 기반 파지(Grasping) |
| Proteus | 2022~ | 자율 이동 로봇 (AMR) | LiDAR + 인간 공존 내비게이션 |
| Sparrow | 2022~ | 개별 상품 피킹 | 다관절 암 + 물체 인식 AI |
| Sequoia | 2023~ | 재고 관리 자동화 | 로봇 간 협업 오케스트레이션 |
| Cardinal | 2023~ | 중량 패키지 분류 | 실시간 무게 추정 + 경로 최적화 |

주목할 점은 이 로봇들이 따로 노는 게 아니라는 겁니다. Sequoia 시스템이 대표적인 사례인데, 여러 로봇 유형이 하나의 워크플로우 안에서 체인처럼 연결되어 돌아갑니다.

![다이어그램 3](/ai-tech-blog/images/posts/2026-02-17/aws와-physical-ai-amazon이-로봇과-물리-세계를-정복하는-법-robomaker부터-100만-/diagram-3.png)

### 100만 대 운용의 아키텍처

Amazon은 2025년 7월 전 세계 풀필먼트 센터에서 100만 대 로봇 배치를 달성했습니다. 300개 이상의 시설에 걸친 이 규모는 로보틱스 역사상 전례 없는 수준입니다. 이 규모의 Fleet Management는 그 자체로 분산 시스템 엔지니어링의 극한입니다. 실제로 써보면 단일 창고 수백 대 수준에서도 스케줄링이 만만치 않은데, 이걸 글로벌 스케일로 돌린다고 생각하면 머리가 아파집니다.

이 규모에서 풀어야 할 핵심 과제를 정리하면 이렇습니다.

1. 경로 충돌 회피 (Multi-Agent Path Finding, MAPF): 수천 대의 로봇이 동시에 이동하는 창고에서 교착(Deadlock) 없이 최적 경로를 실시간으로 계산해야 합니다.
2. 동적 작업 할당 (Dynamic Task Allocation): 주문이 급증하면 로봇 간 워크로드를 밀리초 단위로 재분배해야 합니다.
3. 예측 정비 (Predictive Maintenance): 배터리 열화나 모터 마모를 사전에 감지해서 다운타임을 줄여야 합니다.

간단한 시뮬레이션으로 MAPF 문제의 복잡도를 느껴볼 수 있습니다:

```python
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class State:
    f_cost: int
    position: tuple = field(compare=False)
    time_step: int = field(compare=False)
    path: list = field(compare=False, default_factory=list)

def time_space_astar(grid_size: tuple, start: tuple, goal: tuple,
                     dynamic_obstacles: dict[int, set]) -> list[tuple]:
    """
    시간-공간 A* 탐색: 다른 로봇의 예약된 경로(dynamic_obstacles)를
    시간 축으로 회피하며 최적 경로를 탐색합니다.
    
    Args:
        grid_size: (rows, cols) 창고 그리드 크기
        start: 시작 좌표 (r, c)
        goal: 목표 좌표 (r, c)
        dynamic_obstacles: {time_step: {(r, c), ...}} 시간별 점유 셀
    """
    rows, cols = grid_size
    directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # 대기 포함

    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = [State(f_cost=heuristic(start, goal), position=start, time_step=0, path=[start])]
    visited = set()

    while open_set:
        current = heapq.heappop(open_set)
        if current.position == goal:
            return current.path

        state_key = (current.position, current.time_step)
        if state_key in visited:
            continue
        visited.add(state_key)

        for dr, dc in directions:
            nr, nc = current.position[0] + dr, current.position[1] + dc
            nt = current.time_step + 1
            if 0 <= nr < rows and 0 <= nc < cols:
                next_pos = (nr, nc)
                # 시간 t+1에 다른 로봇이 점유한 셀 회피
                if next_pos not in dynamic_obstacles.get(nt, set()):
                    new_f = nt + heuristic(next_pos, goal)
                    heapq.heappush(open_set, State(
                        f_cost=new_f, position=next_pos,
                        time_step=nt, path=current.path + [next_pos]
                    ))
    return []  # 경로 없음
```

## 시뮬레이션-투-리얼: AWS 기반 Physical AI 개발 파이프라인

실제 로봇을 훈련시키는 건 느리고, 비싸고, 위험합니다. 로봇 팔 하나가 물건을 집는 동작을 학습하는 데 수천 번의 시행착오가 필요한데, 매번 실물로 돌리면 하드웨어가 먼저 망가집니다. AWS가 Physical AI 개발에서 내세우는 원칙은 단순합니다. 시뮬레이션에서 먼저 깨부수고, 현실에선 한 번에 성공시키는 것.

### 디지털 트윈과 합성 데이터 생성

Amazon은 물류 창고 전체를 가상으로 복제(Digital Twin)합니다. 선반 배치, 조명 조건, 컨베이어 벨트 속도, 바닥의 마찰 계수까지 포함한 고충실도(high-fidelity) 환경입니다. 이 가상 창고 안에서 수백 대의 로봇이 동시에 시뮬레이션을 돌리고, 수백만 장의 합성 이미지(Synthetic Data)와 센서 데이터를 뽑아냅니다.

AWS에서는 RoboMaker 시뮬레이션 환경과 Gazebo를 쓸 수 있고, 최근에는 NVIDIA Isaac Sim과 연동해 이 파이프라인을 구축하는 사례도 늘고 있습니다.

![다이어그램 4](/ai-tech-blog/images/posts/2026-02-17/aws와-physical-ai-amazon이-로봇과-물리-세계를-정복하는-법-robomaker부터-100만-/diagram-4.png)

### Sim-to-Real Transfer: 현실과의 간극을 좁히는 법

시뮬레이션에서 완벽하게 동작하던 정책(Policy)이 현실에서 실패하는 현상을 Reality Gap이라 부릅니다. AWS가 제시하는 해법은 크게 두 가지입니다.

첫째, 도메인 랜덤화(Domain Randomization). 시뮬레이션 환경의 물리 파라미터를 매 에피소드마다 무작위로 바꿉니다. 마찰, 질량, 카메라 노이즈, 조명을 극단적으로 흔들어서 모델이 어떤 조건에서도 일반화되도록 강제하는 방식입니다. 개인적으로는 이 접근이 단순하면서도 효과가 확실해서 가장 먼저 시도해볼 만하다고 생각합니다.

```python
import numpy as np

def randomize_domain(env_config: dict) -> dict:
    """시뮬레이션 환경 파라미터를 매 에피소드마다 랜덤화"""
    return {
        "friction": np.random.uniform(0.3, 1.2),
        "object_mass_kg": np.random.uniform(0.05, 5.0),
        "camera_noise_std": np.random.uniform(0.0, 0.05),
        "lighting_intensity": np.random.uniform(0.2, 1.5),
        "gripper_force_offset": np.random.normal(0, 0.1),
        # 텍스처, 색상, 배경도 랜덤화
        "texture_id": np.random.randint(0, env_config["num_textures"]),
    }

# 병렬 시뮬레이션 실행 (AWS RoboMaker에서 수백 개 동시 실행)
for episode in range(100_000):
    params = randomize_domain(base_config)
    env.reset(domain_params=params)
    # ... RL 학습 루프
```

둘째, 점진적 전이(Progressive Transfer). 시뮬레이션에서 곧바로 운영 환경으로 넘어가는 게 아니라, 단순화된 실제 환경을 중간에 두고 단계적으로 이동합니다. 각 단계에서 소량의 실제 데이터로 미세조정(Fine-tuning)하는 구조입니다. Amazon Robotics 내부에서는 이를 "Sim → Lab → Floor" 3단계 파이프라인으로 운영하는 것으로 알려져 있습니다. 실제로 써보면 Lab 단계에서 잡아내는 문제가 꽤 많아서, 이 중간 단계를 건너뛰기 어렵습니다.

### Foundation Model for Robotics: 로봇의 GPT 시대

가장 흥미로운 연구 방향은 로봇 파운데이션 모델(Robot Foundation Model, RFM)입니다. 언어 모델이 대규모 텍스트로 사전학습 후 다양한 태스크에 적용되듯, 로봇 파운데이션 모델도 대규모 로봇 데이터로 사전학습한 뒤 다양한 로봇 플랫폼과 태스크에 범용적으로 적용하는 것을 목표로 합니다. 기존 로봇 학습은 특정 로봇, 특정 환경, 특정 작업에 맞춰 처음부터 훈련해야 했기 때문에 일반화 능력이 극도로 제한적이었습니다. 파운데이션 모델 접근법은 이 한계를 근본적으로 뒤집어, 하나의 거대한 모델이 팔의 자유도가 다른 로봇이든 이동 로봇이든 상관없이 공통된 물리적 직관과 조작 능력을 공유할 수 있도록 설계됩니다.

대표적인 사례로 구글 딥마인드의 RT-2나 오픈소스 진영의 Octo, π₀ 같은 모델을 들 수 있습니다. 이들은 수십만에서 수백만 건의 로봇 에피소드 데이터를 수집하고, 여기에 인터넷에서 얻은 비디오와 텍스트 데이터를 결합하여 학습합니다. 이렇게 훈련된 모델은 자연어 명령을 입력받아 로봇의 행동 시퀀스를 직접 출력할 수 있으며, 학습 과정에서 본 적 없는 새로운 물체나 환경에 대해서도 놀라운 수준의 제로샷 일반화를 보여줍니다.

그러나 로봇 파운데이션 모델이 언어 파운데이션 모델만큼 빠르게 성숙하기에는 여전히 구조적 난관이 존재합니다. 가장 큰 병목은 데이터입니다. 텍스트 데이터는 인터넷에 거의 무한히 존재하지만, 로봇이 실제 물리 환경에서 수집하는 조작 데이터는 수집 비용이 높고 규모를 키우기 어렵습니다. 이를 해결하기 위해 시뮬레이션 환경에서 대량의 합성 데이터를 생성하거나, 사람의 시연 영상에서 행동 정보를 추출하는 연구가 활발히 진행되고 있습니다.

이 분야의 발전 속도는 매우 가파릅니다. 2023년부터 2025년 사이에만 해도 범용 로봇 정책 모델의 성능은 비약적으로 향상되었고, 여러 스타트업과 대형 연구소가 경쟁적으로 데이터 수집 인프라와 모델 아키텍처를 고도화하고 있습니다. 로봇 파운데이션 모델이 충분한 규모와 품질의 데이터를 확보하고 시뮬레이션에서 현실로의 전이 문제를 안정적으로 해결한다면, 하나의 범용 두뇌가 공장의 조립 로봇부터 가정의 서비스 로봇까지 구동하는 시대가 올 것입니다.
## Amazon의 Physical AI 확장 전선 — 물류를 넘어 자율주행과 가정으로

Amazon의 Physical AI 야망은 물류 창고에서 멈추지 않습니다. 2020년 약 12억 달러에 인수한 Zoox는 Amazon이 모빌리티 영역에서 꺼내 든 가장 공격적인 카드입니다.

### Zoox — L5 완전 자율주행이라는 도박

대부분의 자율주행 기업이 L4(특정 구역 내 자율주행)에 집중하는 동안, Zoox는 처음부터 L5(Level 5) 완전 자율주행을 설계 목표로 잡았습니다. 핸들도 페달도 없는 양방향 대칭 차량(bidirectional vehicle)을 자체 설계했는데, 이 지점에서 Waymo나 Cruise와 접근 방식이 근본적으로 갈립니다.

![다이어그램 5](/ai-tech-blog/images/posts/2026-02-17/aws와-physical-ai-amazon이-로봇과-물리-세계를-정복하는-법-robomaker부터-100만-/diagram-5.png)

여기서 눈여겨볼 부분은 AWS 인프라와의 시너지입니다. Zoox는 실도로 주행 데이터를 AWS 클라우드로 전송하고, 수십억 마일 규모의 시뮬레이션을 AWS GPU 클러스터에서 돌립니다. 이 시뮬레이션-투-리얼(Sim-to-Real) 파이프라인은 앞서 살펴본 RoboMaker의 확장판이라 할 수 있습니다.

Zoox는 2025년 9월 라스베이거스에서 완전 무인 로보택시 서비스를 시작했으며, 2026년 초부터 유료 서비스로 전환할 계획입니다. 샌프란시스코 베이 에어리어에서도 같은 해 유료 서비스를 준비 중입니다.

### 가정으로의 침투 — Astro와 그 너머

물류와 도로 다음 전선은 가정입니다. Amazon의 가정용 로봇 Astro는 아직 초기 단계지만, Alexa LLM과 결합하면서 물리 세계를 이해하는 가정용 에이전트 방향으로 진화하고 있습니다. 실제로 써보면 아직 갈 길이 멀다는 느낌이지만, Ring 보안 카메라 생태계와의 연동 덕분에 가정 내 센서 네트워크는 이미 깔려 있는 셈입니다.

```python
# Zoox 스타일의 센서 퓨전 파이프라인 개념 (간소화)
class SensorFusion:
    def __init__(self):
        self.sources = ["lidar_3d", "camera_rgb", "radar_doppler"]

    def fuse(self, sensor_data: dict) -> dict:
        """다중 센서 데이터를 통합 3D 표현으로 변환"""
        point_cloud = sensor_data["lidar_3d"]
        detections = sensor_data["camera_rgb"]  # 2D bbox
        velocities = sensor_data["radar_doppler"]

        # 3D 공간에 투영 후 객체별 상태 벡터 생성
        fused_objects = self._project_and_merge(
            point_cloud, detections, velocities
        )
        return {"objects": fused_objects, "timestamp": sensor_data["ts"]}
```

결국 Amazon의 전략은 명확합니다. 창고에서 도로로, 도로에서 가정으로 이어지는 물리 공간의 연속선 위에서 AWS 클라우드를 두뇌로, 각 영역의 로봇을 신체로 삼아 Physical AI 생태계를 수직 통합하겠다는 것입니다. 개인적으로 이 규모의 시도를 실행할 수 있는 기업은 현재로서는 Amazon 말고는 떠오르지 않습니다.
## References

1. AWS RoboMaker 공식 문서 — 클라우드 로봇 개발, 시뮬레이션 및 플릿 관리 서비스 개요.
   [https://aws.amazon.com/documentation-overview/robomaker/](https://aws.amazon.com/documentation-overview/robomaker/)

2. AWS IoT RoboRunner 소개 (AWS Blog) — 다종 로봇 플릿의 통합 오케스트레이션 및 관리 서비스.
   [https://aws.amazon.com/blogs/aws/preview-aws-iot-roborunner-for-building-robot-fleet-management-applications/](https://aws.amazon.com/blogs/aws/preview-aws-iot-roborunner-for-building-robot-fleet-management-applications/)

3. AWS IoT Greengrass 공식 문서 — 엣지 디바이스에서의 로컬 컴퓨팅, 메시징, ML 추론 지원.
   [https://docs.aws.amazon.com/greengrass/v2/developerguide/what-is-iot-greengrass.html](https://docs.aws.amazon.com/greengrass/v2/developerguide/what-is-iot-greengrass.html)

4. AWS IoT TwinMaker 공식 문서 — 물리 시설 및 로봇 환경의 디지털 트윈 구축 서비스.
   [https://docs.aws.amazon.com/iot-twinmaker/latest/guide/what-is-twinmaker.html](https://docs.aws.amazon.com/iot-twinmaker/latest/guide/what-is-twinmaker.html)

5. Amazon CodeWhisperer / Amazon Q Developer — AI 기반 코드 생성 도구로 ROS 2 로봇 개발 지원.
   [https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/what-is.html](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/what-is.html)

6. Amazon Robotics 소개 — Amazon 물류 센터의 로봇들 (Sparrow, Proteus, Sequoia 등 100만+ 배치).
   [https://www.aboutamazon.com/news/operations/amazon-robotics-robots-fulfillment-center](https://www.aboutamazon.com/news/operations/amazon-robotics-robots-fulfillment-center)

7. MassRobotics Physical AI Fellowship — AWS와 NVIDIA 공동 로보틱스 스타트업 액셀러레이터 프로그램.
   [https://www.therobotreport.com/massrobotics-expands-physical-ai-fellowship-with-aws-and-nvidia/](https://www.therobotreport.com/massrobotics-expands-physical-ai-fellowship-with-aws-and-nvidia/)
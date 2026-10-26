---
title: "AI가 찾은 취약점, 패치가 적용되기까지: PHP 사례로 보는 공개와 대응의 설계"
date: 2026-10-26T10:15:58+09:00
draft: false
author: "Jesam Kim"
description: "PHP2Shell 사례를 통해 AI의 취약점 발견이 실제 보호로 이어지는 과정을 살펴봅니다. 유지관리자의 패치 판단, 공개 일정 조정, 운영 환경의 적용 확인을 함께 정리합니다."
categories: ["AI/ML 기술 심층분석"]
tags: ["AI Security", "OpenAI", "PHP", "Vulnerability Disclosure"]
slug: "php-ai-coordinated-disclosure"
cover:
  image: "/ai-tech-blog/images/php-ai-coordinated-disclosure/cover.png"
  alt: "연결된 도시의 장비를 함께 점검하고 수리하는 엔지니어들"
  relative: false
---

AI가 취약점을 찾아냈다는 소식을 접하면 발견한 모델과 공격의 심각도에 먼저 눈길이 갑니다. 하지만 서비스를 운영하는 입장에서는 그다음이 궁금합니다. 누가 수정 방식을 결정하고, 언제 정보를 공개하며, 내 서버에 수정된 코드가 실행 중이라는 사실은 어떻게 확인할 수 있을까요.

2026년 10월 21일 OpenAI는 <strong>Patch the Planet</strong>의 PHP 대응 사례와 취약점 공개를 조율하는 방안을 발표했습니다. OpenAI의 설명에 따르면 Calif가 GPT-5.6 Sol을 활용해 중대한 결함을 찾았고, Wiz도 두 취약점을 독립적으로 발견해 보고했습니다. 수정과 릴리스는 PHP 유지관리자들이 주도했습니다.

<small>참조: [OpenAI의 Patch the Planet 발표](https://openai.com/index/coordinated-disclosure-of-critical-vulnerabilities/)</small>

이 사례를 읽으며 주목한 부분은 <strong>발견된 취약점을 실제 서비스의 보호로 바꾸는 과정</strong>입니다. 공개 자료를 함께 읽으면 AI를 활용한 연구, 유지관리자의 판단, 배포판의 준비, 운영자의 적용 확인이 서로 다른 일이라는 점이 드러납니다. 이 글은 2026년 10월 26일까지의 공지를 기준으로 그 과정과 운영상의 의미를 정리합니다.

## 발견과 수정에는 서로 다른 책임이 있습니다

PHP2Shell로 알려진 두 취약점은 PHP 엔진의 메모리 안전성 문제입니다. CVE-2026-109618은 메모리 부족 상황에서 불완전한 요청 상태를 정리하는 과정과 관련됩니다. CVE-2026-109619는 해제된 요청 메모리와 연결된 상태가 남아 후속 요청에서 사용되는 문제를 포함합니다. PHP 자체의 문제이므로 애플리케이션 코드를 잘 작성했다는 사실만으로 영향을 배제할 수는 없습니다.

다만 실행 구성에 따라 공격 조건과 검증 범위가 다릅니다. 공식 자문에서 첫 번째 CVE의 원격 코드 실행은 mod_php에서 확인됐지만 FPM의 완전한 공격 경로는 검증되지 않았다고 설명합니다. 두 번째 CVE에는 mod_php와 FPM에서 검증한 세션 관련 경로가 있으며, 애플리케이션 동작과 같은 워커로의 요청 전달 등 조건이 붙습니다. 이를 모든 PHP 구성에서 동일한 공격이 입증된 것으로 읽으면 범위가 달라집니다. 여기서는 실행 구성별 검증 범위를 상세 GHSA 자문에 맞췄습니다. PHP 재단 회고의 FPM 요약과는 설명에 차이가 있습니다.

<small>참조: [CVE-2026-109618 공식 자문](https://github.com/php/php-src/security/advisories/GHSA-w8hw-w9fx-j79w), [CVE-2026-109619 공식 자문](https://github.com/php/php-src/security/advisories/GHSA-6wmf-4rv5-p3rp)</small>

Wiz는 복잡한 공격 경로를 구성하고 대상 환경에 맞게 시험하는 데 AI가 도움을 줬다고 설명합니다. 이 결과는 결함이 실제로 얼마나 위험한지 판단하는 근거가 됩니다. 하지만 그 근거만으로 어떤 수정을 배포할지 자동으로 결정되지는 않습니다.

<small>참조: [Wiz의 PHP2Shell 기술 설명](https://www.wiz.io/blog/php2shell-php-rce)</small>

PHP 재단의 회고에 따르면 메모리 부족 문제의 초기 징후는 2024년에 발견됐습니다. 수정 후보마다 하위 호환성, 서비스 거부 가능성, 외부 확장 모듈 변경에 대한 부담이 있었고, 의사결정 방식과 전문 인력 부족으로 해결이 지연됐습니다. 2026년 Calif 등의 보고가 실제 영향을 구체화한 뒤 수정이 최우선 과제가 됐습니다. 재단은 이 지연을 구조적인 실패로 인정하고 의사결정 절차를 바꾸고 있다고 밝혔습니다.

<small>참조: [PHP 재단의 보안 릴리스 회고](https://thephp.foundation/blog/2026/10/21/todays-php-security-release-and-how-were-strengthening-php-security/)</small>

![연구자는 재현과 영향 분석을, 유지관리자는 수정과 릴리스를, 공급자는 배포 준비를, 운영자는 적용 확인을 담당합니다.](/ai-tech-blog/images/php-ai-coordinated-disclosure/roles.png)

*공개 자료의 역할을 단순화한 개념도입니다. 단계 간 협업을 설명하며, 모든 조직이 이 순서로만 일했다는 뜻은 아닙니다.*

<small>참조: [OpenAI 발표](https://openai.com/index/coordinated-disclosure-of-critical-vulnerabilities/), [PHP 재단 회고](https://thephp.foundation/blog/2026/10/21/todays-php-security-release-and-how-were-strengthening-php-security/)</small>

여기서 얻을 수 있는 설계상의 교훈은 분명합니다. AI로 보고서를 더 많이 만드는 일과 유지관리자가 결정을 내릴 수 있게 돕는 일을 함께 계획해야 합니다. 재현 가능한 증거, 회귀 테스트, 호환성 검토, 해결이 지연될 때 판단을 요청할 책임자가 필요합니다. 이는 이 사례를 바탕으로 정리한 운영 제안입니다.

## 공개 전 준비 시간에도 공격은 시작될 수 있습니다

조율된 취약점 공개는 유지관리자와 관련 공급자가 패치를 준비할 시간을 확보하는 절차입니다. 참여자를 늘리면 더 많은 환경을 보호할 수 있지만, 공개 전 정보를 다루는 조직도 늘어납니다.

OpenAI의 제안은 유지관리자가 참여 조직을 결정하고, 각 조직의 보호 역할과 준비 시간, 민감한 정보를 다룰 능력을 고려하도록 합니다. 필요한 범위의 정보를 공유하되, 악용이나 유출, 취약점을 드러내는 공개 패치 등이 나타나면 일정을 재검토하는 방식입니다. 아직 보편적인 표준으로 확정된 절차가 아닌 <strong>공개적으로 제안된 대응 틀</strong>입니다.

<small>참조: [OpenAI가 제안한 조율 방식](https://openai.com/index/coordinated-disclosure-of-critical-vulnerabilities/)</small>

실제 PHP 대응은 예정대로만 진행되지 않았습니다. PHP 재단은 10월 20일 CVE 식별자 유출에 대응해, 원래 22일로 잡았던 릴리스를 21일로 앞당겼다고 밝혔습니다. 이후 사전 공지 뒤에 시연 영상도 잘못 공개됐습니다. 즉시 공개하지 않은 이유로는 배포판들이 수정 패키지를 준비 중이었다는 점을 들었습니다. 공개 속도와 사용자가 실제로 업데이트할 수 있는 시점 사이에서 판단이 필요했던 것입니다.

<small>참조: [PHP 재단의 공개 일정 변경 설명](https://thephp.foundation/blog/2026/10/21/todays-php-security-release-and-how-were-strengthening-php-security/)</small>

별도로 AWS는 MadPot 관측 기록에서 10월 17일의 PHP2Shell 공격 시도를 확인했다고 발표했습니다. 공개일보다 나흘 앞선 시도입니다. 다만 AWS는 해당 시도들이 실제 운영 시스템에서 성공했을지는 불분명하며, 공격자가 공개 전 정보를 어떻게 얻었는지도 알 수 없다고 명시했습니다. <strong>공격 시도의 관측과 실제 침해의 확인은 구분해야 합니다.</strong> 이 자료만으로 해당 공격자가 AI를 사용했다고 단정할 수도 없습니다.

<small>참조: [AWS의 공개 전 공격 시도 관측](https://aws.amazon.com/blogs/security/amazon-threat-intelligence-identifies-php2shell-exploitation-prior-to-public-disclosure-in-madpot/)</small>

![10월 17일 공격 시도 관측부터 20일 사전 공지, 21일 보안 릴리스, 23일 추가 공지까지 이어진 대응 일정입니다.](/ai-tech-blog/images/php-ai-coordinated-disclosure/timeline.png)

*발생일 기준으로 공개 기록을 재구성했습니다. 17일의 시도는 AWS가 21일 발표에서 보고한 내용이며, 침해 성공을 뜻하지 않습니다.*

<small>참조: [AWS 관측 보고](https://aws.amazon.com/blogs/security/amazon-threat-intelligence-identifies-php2shell-exploitation-prior-to-public-disclosure-in-madpot/), [PHP 공식 공지](https://www.php.net/archive/2026.php), [PHP 재단 회고](https://thephp.foundation/blog/2026/10/21/todays-php-security-release-and-how-were-strengthening-php-security/)</small>

따라서 운영자가 조사할 기간을 공개일 이후로만 제한하면 놓치는 정보가 생길 수 있습니다. 공지를 받은 시점에는 업데이트 준비와 함께, 확보한 로그에서 더 이른 이상 징후를 확인할 필요가 있습니다. 조사 범위는 자신의 노출 기간과 보존된 기록에 맞춰 정해야 합니다.

## 임시 보호와 패치 적용을 따로 확인해야 합니다

Cloudflare는 공개 전에 WAF 보호를 준비해 배포했다고 밝혔습니다. 이런 조치는 사용자가 패키지를 교체하는 동안 위험을 줄이는 데 도움이 됩니다. 그러나 네트워크에서 특정 요청을 차단하는 조치와 PHP 실행 코드의 결함을 수정하는 조치는 역할이 다릅니다.

이 차이는 후속 공지에서도 드러납니다. Cloudflare는 10월 23일 관련 취약점에 대해 알려진 공격 시도를 막는 규칙을 설명하면서, PHP의 기본 업로드 진행 필드 이름을 바꾼 사이트는 해당 규칙의 보호 대상에 포함되지 않는다고 밝혔습니다. 따라서 WAF 사용 여부만 확인해서 조치를 완료 처리하면 적용 조건을 놓칠 수 있습니다.

<small>참조: [Cloudflare의 보호 조치와 10월 23일 업데이트](https://blog.cloudflare.com/php-vulnerabilities-waf-protection/)</small>

10월 21일 두 CVE의 공식 수정 버전은 다음과 같습니다. 배포판 패키지를 쓰는 경우에는 상위 PHP 버전만 비교하지 말고 공급사의 보안 공지와 패키지 릴리스를 함께 확인해야 합니다. 이전 버전 번호를 유지하면서 수정 사항을 반영하는 백포트가 있을 수 있기 때문입니다.

| PHP 브랜치 | 두 CVE의 최초 수정 버전 |
|---|---|
| 8.2 | 8.2.35 |
| 8.3 | 8.3.36 |
| 8.4 | 8.4.27 |
| 8.5 | 8.5.12 |

<small>참조: [PHP 10월 21일 보안 공지](https://www.php.net/archive/2026.php#2026-10-21-5), [CVE-2026-109618 자문](https://github.com/php/php-src/security/advisories/GHSA-w8hw-w9fx-j79w), [CVE-2026-109619 자문](https://github.com/php/php-src/security/advisories/GHSA-6wmf-4rv5-p3rp)</small>

여기에는 후속 확인이 하나 더 필요합니다. PHP는 10월 23일 특정 세션 설정 조합에서 남는 원격 코드 실행 가능성을 별도 CVE-2026-115560으로 안내했습니다. 설정의 serializer가 `php_binary` 또는 `php_serialize`이고, 실행 중 `session_start()` 전에 `php`로 변경하며 업로드 진행 기능이 활성화된 경우입니다. 공지는 다음 정기 수정판까지 해당 조건에서 `session.upload_progress.enabled=0`을 사용하도록 안내합니다. 실행 중 serializer를 변경하지 않는 애플리케이션은 기능을 유지할 수 있다고 설명합니다. 26일 확인한 이 공지에는 전체 자문과 수정 버전이 아직 제시되지 않았습니다.

<small>참조: [PHP 10월 23일 추가 공지](https://www.php.net/archive/2026.php#2026-10-23-1)</small>

이는 최초 두 CVE의 수정과 구분해야 하는 후속 문제입니다. 패치를 적용한 뒤에도 자신에게 해당하는 추가 조건과 새 공지를 추적하는 담당자가 필요합니다.

## 완료 기준은 실행 중인 환경에 있어야 합니다

OpenAI가 공개한 PHP 버전 검사 도구는 대응 대상을 찾는 데 쓸 수 있습니다. URL 검사에서는 HTTP 헤더가 광고하는 PHP 버전을 기준으로 경고합니다. README는 `OK`도 설치 상태를 검증하는 결과가 아니라고 명시합니다. 숨겨진 헤더는 `UNKNOWN`이 되며, 헤더만으로 백포트 적용 여부를 판단할 수 없습니다. 공격 코드를 보내 취약성을 검증하는 도구도 아닙니다.

<small>참조: [OpenAI PHP Version Checker README](https://github.com/openai/php2shell-checker)</small>

따라서 이 도구의 결과는 자산 조사에 쓰는 단서로 다루는 편이 적절합니다. 다음은 공개 자료를 바탕으로 이 글에서 제안하는 운영 절차입니다.

![영향 환경 식별, 공식 패치 선택, 배포와 재시작, 실제 실행 상태 확인, 후속 공지 추적을 거쳐 대응을 관리합니다.](/ai-tech-blog/images/php-ai-coordinated-disclosure/verification.png)

*이 글에서 제안하는 적용 확인 절차입니다. PHP 프로젝트가 그대로 운영했다는 의미는 아니며, 침해 조사와 임시 완화책은 업데이트와 병행할 수 있습니다.*

<small>참조: [PHP 공식 공지](https://www.php.net/archive/2026.php), [버전 검사 도구의 제한](https://github.com/openai/php2shell-checker)</small>

1. <strong>실제로 요청을 처리하는 환경을 찾습니다.</strong> 사이트별 PHP 실행 방식, 패키지 공급자, 컨테이너 이미지와 담당자를 기록합니다. 한 서버의 CLI 버전이 웹 요청을 처리하는 PHP와 같은지도 확인합니다.
2. <strong>공급사 공지로 수정 여부를 판단합니다.</strong> 위 표의 상위 릴리스 또는 공급사의 백포트 공지를 근거로 대상 패키지를 선택합니다. 지원이 끝난 버전은 별도 이전 계획도 세웁니다.
3. <strong>배포 후 실행 상태를 확인합니다.</strong> 패키지 설치나 이미지 빌드가 끝난 시각과 실제 워커 교체가 끝난 시각을 구분합니다. 서비스 방식에 맞는 재시작이나 재배포 후 기능 점검과 오류 관측을 수행합니다.
4. <strong>완화책과 조사 결과를 따로 남깁니다.</strong> WAF 규칙의 적용 조건과 예외, 침해 조사 필요 여부, 아직 바꾸지 못한 자산을 기록합니다. 임시 조치에는 영구 수정으로 전환할 담당자와 일정을 붙입니다.
5. <strong>후속 공지를 추적합니다.</strong> 추가 취약점이나 적용 조건이 나오면 완료한 자산도 다시 검토할 수 있어야 합니다.

예를 들어 컨테이너 기반 서비스라면 “수정된 이미지를 만들었다”는 기록에 더해, 트래픽을 받는 인스턴스들이 해당 이미지로 교체됐는지를 확인해야 합니다. 이 예시는 특정 PHP 사고의 조사 결과가 아닌 운영 절차를 설명하기 위한 것입니다.

PHP 사례를 통해 AI 보안의 성과를 평가할 때 함께 볼 항목이 생깁니다. 발견한 취약점 수에 더해, 유지관리자가 수정 결정을 내리는 데 걸린 시간, 공급자가 업데이트를 준비한 범위, 실제 서비스에 수정이 적용되기까지 걸린 시간을 확인해야 합니다. <strong>취약점을 찾는 능력이 빨라질수록, 그 결과를 검증하고 배포하는 사람과 절차에도 투자해야 합니다.</strong>

## References

- [OpenAI: Patch the Planet과 공개 대응 제안](https://openai.com/index/coordinated-disclosure-of-critical-vulnerabilities/)
- [PHP Foundation: 보안 릴리스와 의사결정 및 공개 과정 회고](https://thephp.foundation/blog/2026/10/21/todays-php-security-release-and-how-were-strengthening-php-security/)
- [PHP: 2026년 공식 공지](https://www.php.net/archive/2026.php)
- [PHP: CVE-2026-109618 공식 자문](https://github.com/php/php-src/security/advisories/GHSA-w8hw-w9fx-j79w)
- [PHP: CVE-2026-109619 공식 자문](https://github.com/php/php-src/security/advisories/GHSA-6wmf-4rv5-p3rp)
- [Wiz: PHP2Shell 기술 설명](https://www.wiz.io/blog/php2shell-php-rce)
- [AWS: 공개 전 공격 시도 관측](https://aws.amazon.com/blogs/security/amazon-threat-intelligence-identifies-php2shell-exploitation-prior-to-public-disclosure-in-madpot/)
- [Cloudflare: WAF 보호와 적용 조건](https://blog.cloudflare.com/php-vulnerabilities-waf-protection/)
- [OpenAI: PHP Version Checker](https://github.com/openai/php2shell-checker)


## "백업" 명령어
Jay가 "백업 해" 라고 하면:
1. 변경된 파일 모두 `git add -A`
2. `git commit` (변경 내용 요약)
3. `git push` to https://github.com/jesamkim/clawd (private)

## Gmail 사용 방법 (gog CLI 대신 Python 사용)

gog CLI는 현재 OAuth 인증 문제로 사용 불가.
대신 Python 스크립트로 Gmail을 확인해야 함:

```
python3 ~/google-helper/gmail_check.py
```

이 스크립트는 최근 메일 5개를 조회함.
Gmail 관련 요청이 오면 위 스크립트를 exec로 실행할 것.

# Git/GitHub + 실습환경 가이드

이 문서는 이 저장소에서 강의자료 작성과 실습 코드 관리를 함께 진행할 때 필요한 최소한의 작업 규칙을 정리한다.

## 0. 이 저장소의 역할

- `docs/`: 기존 원고 또는 참고용 소스 문서
- `lecture/`: 최종 강의자료 원고
- `practice/`: 장별 실습 코드와 데이터

핵심 원칙은 다음과 같다.

1. `docs/`는 참고 원고로 유지한다.
2. `lecture/`는 `causal/lecture` 스타일에 맞춰 새로 작성한다.
3. 강의자료와 실습 코드는 장 번호를 기준으로 대응시킨다.

## 1. 기본 작업 흐름

권장 작업 순서는 아래와 같다.

```bash
git status
# 문서 또는 코드 수정
git add -A
git commit -m "chapter04: update lecture scaffold"
git push
```

문서 작성 전에는 가능하면 아래를 먼저 확인한다.

```bash
git pull
git status
```

## 2. 현재 표준 구조

```text
.
├─ docs/                  # 기존 장별 원고
├─ lecture/               # 새 강의자료
│  ├─ README.md
│  ├─ STYLE_GUIDE.md
│  └─ TEMPLATE.md
├─ practice/
│  ├─ chapter01/
│  ├─ chapter02/
│  ├─ ...
│  └─ chapter12/
├─ .gitignore
├─ github.md
├─ requirements.txt
├─ requirements-optional.txt
└─ setup_env.py
```

## 3. 문서 작성 원칙

- `docs/` 내용을 그대로 복사하지 않는다.
- `lecture/` 문서는 학부생 대상 강의자료 톤으로 다시 쓴다.
- 각 장은 장 제목, 학습 목표, 미리보기, 본문 절, 핵심 요약, 실습 연결을 포함한다.
- 실습 참조 경로는 `practice/chapterXX/...` 형식으로 통일한다.

세부 규칙은 `lecture/STYLE_GUIDE.md`와 `lecture/TEMPLATE.md`를 따른다.

## 4. 실습 환경 준비

프로젝트 루트에서 아래 명령으로 기본 환경을 만들 수 있다.

```bash
python setup_env.py
```

설치 확인만 하고 싶다면:

```bash
python setup_env.py --check
```

고급 장에 필요한 선택 패키지는 아래처럼 추가 설치한다.

```bash
python setup_env.py --extras llm
python setup_env.py --extras torch
python setup_env.py --extras graph
```

## 5. 주의할 점

- `practice/**/venv`, `tmpclaude-*`, `.DS_Store`는 버전 관리 대상이 아니다.
- 장 번호는 `chapter01` 형식으로 통일한다.
- `lecture/` 원고 작성 전에는 먼저 장별 개요와 실습 연결 포인트를 정리한다.

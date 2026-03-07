# 의사결정나무 예시: 대출 승인 (Mermaid 소스)

이 파일은 `diagram/3-3.png`로 렌더링되어야 합니다.

```mermaid
graph TD
    A["<b>루트 노드</b><br/>전체 12명<br/>승인 6 / 거절 6<br/><i>지니 = 0.50</i>"]

    A -->|"연소득 ≤ 4천만원"| B["<b>노드 B</b><br/>6명<br/>승인 1 / 거절 5<br/><i>지니 = 0.28</i>"]
    A -->|"연소득 > 4천만원"| C["<b>노드 C</b><br/>6명<br/>승인 5 / 거절 1<br/><i>지니 = 0.28</i>"]

    B -->|"신용점수 ≤ 650"| D["<b>리프 D</b><br/>4명<br/>거절 100%<br/><i>지니 = 0.00</i>"]
    B -->|"신용점수 > 650"| E["<b>리프 E</b><br/>2명<br/>혼합 50%/50%<br/><i>지니 = 0.50</i>"]

    C -->|"신용점수 ≤ 700"| F["<b>리프 F</b><br/>2명<br/>혼합 50%/50%<br/><i>지니 = 0.50</i>"]
    C -->|"신용점수 > 700"| G["<b>리프 G</b><br/>4명<br/>승인 100%<br/><i>지니 = 0.00</i>"]

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#ffe6e6,stroke:#c00
    style C fill:#e6ffe6,stroke:#0a0
    style D fill:#ff9999,stroke:#c00
    style E fill:#ffff99,stroke:#cc0
    style F fill:#ffff99,stroke:#cc0
    style G fill:#99ff99,stroke:#0a0
```

## 렌더링 방법

1. Mermaid Live Editor: https://mermaid.live
2. VS Code Mermaid 확장 사용
3. mmdc CLI: `mmdc -i 3-3-decision-tree-loan.md -o 3-3.png`

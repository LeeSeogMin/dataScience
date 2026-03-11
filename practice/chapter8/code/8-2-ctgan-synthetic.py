"""
8.2 CTGAN 합성 데이터 생성 (선택 실행)

주의:
- 이 스크립트는 `ctgan` 패키지 설치가 필요하다.
- 현재 저장소의 기본 requirements에는 포함되어 있지 않을 수 있으므로, 설치되지 않은 경우 안내 메시지만 출력하고 종료한다.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"


def main() -> None:
    if not find_spec("ctgan"):
        print("ctgan 패키지가 설치되어 있지 않습니다.")
        print("이 스크립트는 선택 실습이며, 설치 후에만 실행할 수 있습니다.")
        print("예) pip install ctgan")
        return

    from ctgan import CTGAN  # type: ignore

    df = pd.read_csv(INPUT_DIR / "adult_income.csv")

    # 예시: 범주형 컬럼을 포함한 경우, discrete_columns에 지정한다.
    discrete_columns = ["marital_status", "workclass", "income"]

    ctgan = CTGAN(epochs=300, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_columns)

    synthetic = ctgan.sample(len(df))
    print(synthetic.head().to_string(index=False))


if __name__ == "__main__":
    main()


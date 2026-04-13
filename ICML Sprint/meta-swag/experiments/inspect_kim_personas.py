from __future__ import annotations

from pathlib import Path

import pandas as pd

from meta_swag.kim_reference import (
    load_ipd_personas,
    load_rps_personas,
    summarize_persona_bundle,
)


def main() -> None:
    output_dir = Path("experiments/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for loader in (load_ipd_personas, load_rps_personas):
        bundles = loader(split="test")
        for bundle in bundles.values():
            rows.append(summarize_persona_bundle(bundle))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "kim_persona_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved persona summary to {output_dir / 'kim_persona_summary.csv'}")


if __name__ == "__main__":
    main()


"""Compare a reproduced 18-window run against archived aggregate scores."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scores", type=Path, default=Path("outputs/evaluation/scores.csv"))
parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/reference_scores.csv")
args = parser.parse_args()
keys = ["model", "window"]
actual = pd.read_csv(args.scores).set_index(keys).sort_index()
expected = pd.read_csv(args.reference).set_index(keys).sort_index()
assert actual.index.is_unique and expected.index.is_unique
assert actual.index.equals(expected.index), "Expected all 72 model/window comparisons"
np.testing.assert_allclose(actual[["combined", "diag", "off"]], expected[["combined", "diag", "off"]], atol=1e-8, rtol=1e-7)
print("PASS: all 72 model/window comparisons match the archived implementation.")

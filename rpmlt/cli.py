"""Command-line entry points; no API key, hidden data or shared repo required."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .data import load_data, submission_dates
from .evaluation import evaluate
from .models import predict
from .submission import validate, write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Reproduce the main comparison table")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluation.add_argument("--protocol", choices=["all", "diagnostic", "legacy"], default="all")
    submission = commands.add_parser("predict", help="Generate, but do not upload, a submission TSV")
    submission.add_argument("--data", type=Path, required=True)
    submission.add_argument("--output", type=Path, default=Path("outputs/submission.tsv"))
    submission.add_argument("--validator", type=Path, help="Path to independently obtained official validator")
    validation = commands.add_parser("validate", help="Check dates, grid IDs and finite nonnegative values")
    validation.add_argument("submission", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate(args.submission)))
        return
    data = load_data(args.data)
    if args.command == "evaluate":
        evaluate(data, args.output, args.protocol)
        return
    result = write_submission(args.output, args.data, data, predict(data, submission_dates()))
    if args.validator:
        proc = subprocess.run([sys.executable, str(args.validator), str(args.output)],
                              capture_output=True, text=True, check=False)
        print(proc.stdout)
        if proc.returncode or "Validation passed!" not in proc.stdout:
            raise RuntimeError(f"Official validator failed: {proc.stderr}")
        result["official_validation"] = "Validation passed!"
    else:
        result["official_validation"] = "not run; local validation is not a substitute"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

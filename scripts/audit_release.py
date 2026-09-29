"""Audit exactly the Git index before publishing, without echoing secret values."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [p for p in paths if p]
    allowed_roots = {"rpmlt", "scripts", "tests", "benchmarks", ".github"}
    allowed_files = {"README.md", "DATA_SOURCES.md", "REPRODUCIBILITY.md", "LICENSE",
                     "pyproject.toml", "requirements-repro.txt", ".gitignore"}
    allowed_data = {"rpmlt/assets/cells.csv", "rpmlt/assets/water_daily.csv", "rpmlt/assets/holidays.csv",
                    "rpmlt/assets/nanao_water.csv", "rpmlt/assets/festivals.json",
                    "rpmlt/assets/source_manifest.json", "benchmarks/reference_scores.csv"}
    patterns = [re.compile(r"sk-" + r"[A-Za-z0-9_-]{20,}"),
                re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{20,}"),
                re.compile(r"github_pat_" + r"[A-Za-z0-9_]{20,}"),
                re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
                re.compile(r"-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
    issues, total = [], 0
    for name in paths:
        path = Path(name)
        if name not in allowed_files and path.parts[0] not in allowed_roots:
            issues.append((name, "outside release allowlist"))
        if path.suffix.lower() in {".csv", ".json"} and name not in allowed_data:
            issues.append((name, "unapproved data file"))
        if path.suffix.lower() in {".tsv", ".npz", ".npy", ".parquet", ".pkl", ".pdf", ".png", ".zip"}:
            issues.append((name, "restricted/generated/binary artifact"))
        body = subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)
        total += len(body)
        if len(body) > 1_000_000:
            issues.append((name, "unexpectedly large file"))
        text = body.decode("utf-8")
        if any(pattern.search(text) for pattern in patterns):
            issues.append((name, "possible credential; value withheld"))
        if re.search(r"[A-Za-z]:[/\\](?:Users|work)[/\\]", text):
            issues.append((name, "machine-specific workspace path"))
    if issues:
        raise SystemExit(str(issues))
    print(f"PASS: {len(paths)} indexed text files, {total:,} bytes; no disallowed data or detected credentials.")


if __name__ == "__main__":
    main()

"""Scan the candidate without importing or running research code."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-public-approval", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "runs"}
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and not any(s in skip or s.endswith(".egg-info") for s in p.relative_to(root).parts)]
    errors = []
    patterns = {
        "private key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}",
        "AWS credential": r"AKIA[0-9A-Z]{16}",
        "assigned secret": r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{8,}[\x22\x27]",
        "personal machine path": r"/(?:home|Users)/[^\s`]+",
    }
    for p in files:
        name = str(p.relative_to(root))
        if p.is_symlink() or p.stat().st_size > 1024 * 1024:
            errors.append(f"{name}: symlink or oversized candidate file")
            continue
        if p.suffix.lower() in {".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".zip", ".gz", ".pdf", ".png", ".svg"}:
            errors.append(f"{name}: private/binary/generated payload")
            continue
        try:
            text = p.read_text()
        except UnicodeError:
            errors.append(f"{name}: non-text payload")
            continue
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                errors.append(f"{name}: {label}")
        if p.suffix == ".csv" and name != "results/reconstruction-2026-10-03/metrics.csv":
            errors.append(f"{name}: unauthorized CSV")
        if p.suffix == ".ipynb":
            nb = json.loads(text)
            if any(c.get("outputs") or c.get("execution_count") is not None for c in nb["cells"] if c["cell_type"] == "code"):
                errors.append(f"{name}: historical notebook still contains outputs")
        if p.suffix == ".json" and '"learner"' in text:
            errors.append(f"{name}: tree model payload")
    # Scan staged/tracked Git objects too; ignored status cannot hide a staged payload.
    git = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if git.returncode == 0:
        allowed = {str(p.relative_to(root)) for p in files}
        for name in git.stdout.decode().split("\0"):
            if name and name not in allowed:
                errors.append(f"{name}: tracked payload excluded from candidate scan")
    status = json.loads((root / "docs" / "PUBLICATION_STATUS.json").read_text())
    if args.require_public_approval:
        if status.get("github_publication_authorized_by_user") is not True:
            errors.append("Publication hold: explicit user upload instruction absent")
        if status.get("project_code_license") and status.get("project_code_license_authorized_by_rights_holders") is not True:
            errors.append("Publication hold: blanket code license has no recorded rights-holder approval")
    print(json.dumps({"files_checked": len(files), "errors": errors,
                      "scope": "Candidate tree and tracked file names; no original Git history exists",
                      "limitation": "Pattern scan is bounded; the upload-instruction gate does not establish third-party rights"}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

"""Exit 0 when every package pinned in requirements.txt is installed at that version.
Used by run_all.ps1 to skip pip (and the network) on reruns."""
import sys
from importlib import metadata
from pathlib import Path


def main() -> int:
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    bad = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, want = (x.strip() for x in line.split("==", 1))
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != want:
            bad.append(f"{name} {have or 'missing'} (need {want})")
    for b in bad:
        print("package to install:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

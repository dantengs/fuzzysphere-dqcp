"""SHA-256 manifest of a directory of frozen analysis files.

    python analysis/tools/manifest.py write MANIFEST.sha256 FILE...   (paths relative to the manifest's directory)
    python analysis/tools/manifest.py check MANIFEST.sha256

Lines are "<sha256>  <relative path>"; the hash is of the raw bytes (analysis/.gitattributes keeps them unchanged).
"""
import hashlib
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(manifest, files):
    root = Path(manifest).resolve().parent
    lines = [f"{sha(root / f)}  {Path(f).as_posix()}\n" for f in files]
    Path(manifest).write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {manifest}: {len(lines)} files")


def check(manifest):
    root = Path(manifest).resolve().parent
    bad = 0
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        p = root / rel.strip()
        if not p.is_file() or sha(p) != digest:
            print("DIFFERS", rel.strip())
            bad += 1
    print("OK" if bad == 0 else f"{bad} file(s) differ")
    return bad


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "write":
        write(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        sys.exit(1 if check(sys.argv[2]) else 0)
    else:
        sys.exit(__doc__)

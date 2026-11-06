#!/usr/bin/env python3
"""Canonical content digests, byte-identical to Tcc digest.jl (file_digest / tree_digest).

file : SHA-256 of the bytes with CRLF -> LF, unless the file contains a NUL byte (binary, raw).
tree : files under DIR (directories named .git skipped), POSIX relative paths sorted by their
       UTF-8 bytes; each contributes  <path> 0x00 <file digest hex> 0x0a  to one SHA-256.
usage: canonical_hash.py file PATH | tree DIR
"""
import hashlib, os, sys


def _lf(b: bytes) -> bytes:
    return b if b"\x00" in b else b.replace(b"\r\n", b"\n")


def file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(_lf(f.read())).hexdigest()


def tree_digest(root: str) -> str:
    rels = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            rels.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    rels.sort(key=lambda r: r.encode("utf-8"))
    h = hashlib.sha256()
    for r in rels:
        h.update(r.encode("utf-8") + b"\x00" + file_digest(os.path.join(root, r)).encode() + b"\n")
    return h.hexdigest()


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    print(file_digest(path) if kind == "file" else tree_digest(path))

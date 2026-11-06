"""Formal driver (v0.5.0 rules; v0.5.1 code).  python -X utf8 -B -m blindpd.formal <config.json> <out_dir>

Runs the three frozen steps into <out_dir>/run, <out_dir>/harden, <out_dir>/resolve:
  1. blindpd.run       spectrum, families, raw-level classes with descendant_span_v1 (cfg['descendant_span'] required)
  2. blindpd.harden    harden_v1 checks on that run (config/harden_v1.json with the baseline paths filled in)
  3. blindpd.resolve_s3  S3 resolution and S4 propagation
The harden configuration actually used is written to <out_dir>/harden_config.json.  Nothing else is changed.
"""
import json, os, sys
from . import run as run_mod, harden as harden_mod, resolve_s3


def main(cfg_path, out):
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"refusing to write into non-empty {out}")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    if not cfg.get("descendant_span") or cfg["descendant_span"].get("option") != "B":
        sys.exit("formal runs require descendant_span option B (config/descendant_span_v1.json, S0)")
    os.makedirs(out, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir, hard_dir, res_dir = (os.path.join(out, d) for d in ("run", "harden", "resolve"))
    run_mod.main(cfg_path, run_dir)
    hc = json.load(open(os.path.join(root, "config", "harden_v1.json"), encoding="utf-8"))
    hc["run_id"] = f"harden_v1_on_{cfg.get('run_id', 'formal')}"
    hc["baseline"] = dict(config=os.path.abspath(cfg_path).replace(os.sep, "/"), run=os.path.abspath(run_dir).replace(os.sep, "/"),
                          code_manifest="CODE_MANIFEST_v0.5.1.sha256",
                          note="filled in by blindpd.formal; harden_v1 rules unchanged")
    hpath = os.path.join(out, "harden_config.json")
    json.dump(hc, open(hpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    harden_mod.main(hpath, hard_dir)
    resolve_s3.main(os.path.abspath(run_dir), os.path.abspath(hard_dir), res_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

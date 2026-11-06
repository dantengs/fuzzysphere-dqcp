"""Fill the data-root placeholders of a template config (rule A) with the paths of your machine.

    python analysis/tools/localize.py TEMPLATE.json PATHS.json OUT.json

TEMPLATE.json is a config whose input paths start with ${NAME} (analysis/hardgate/config/formal.template.json);
PATHS.json maps NAME -> absolute directory or file (copy analysis/paths.example.json to analysis/paths.local.json,
which git ignores).  ${ANALYSIS} is always this analysis/ directory.  Only path strings change: every other key
of the config is written back unchanged.  Unknown placeholders are an error.

Write OUT.json into a directory <base>/config/ on the same drive as the inputs: blindpd resolves relative paths
against <base> and compares paths with os.path.relpath, which fails across Windows drives.
"""
import json
import re
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent.parent
PH = re.compile(r"\$\{([A-Z0-9_]+)\}")


def fill(x, roots):
    if isinstance(x, dict):
        return {k: fill(v, roots) for k, v in x.items()}
    if isinstance(x, list):
        return [fill(v, roots) for v in x]
    if isinstance(x, str):
        def sub(m):
            name = m.group(1)
            if name not in roots:
                sys.exit(f"placeholder ${{{name}}} has no entry in the paths file")
            return roots[name].rstrip("/\\").replace("\\", "/")
        return PH.sub(sub, x)
    return x


def main(template, paths, out):
    roots = json.loads(Path(paths).read_text(encoding="utf-8"))
    roots = {k: v for k, v in roots.items() if not k.startswith("_")}
    roots["ANALYSIS"] = str(ANALYSIS)
    cfg = fill(json.loads(Path(template).read_text(encoding="utf-8")), roots)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])

import re
import sys
from pathlib import Path


LAYER_RE = re.compile(
    r"Layer\s+(\d+)\s+\[(attn|ssm)\]:\s+hidden_rms=([0-9.]+)\s+branch_rms=([0-9.]+)"
)
TOP_RE = re.compile(r"Top-5 logits:(.*)")
PAIR_RE = re.compile(r"\[(\d+)\]=(-?[0-9.]+)")


def parse_trace(path: Path):
    layers = {}
    top5 = []
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace").splitlines()
    else:
        text = raw.decode("utf-8", errors="replace").splitlines()
    for line in text:
        m = LAYER_RE.search(line)
        if m:
            layer = int(m.group(1))
            layers[layer] = {
                "kind": m.group(2),
                "hidden_rms": float(m.group(3)),
                "branch_rms": float(m.group(4)),
            }
            continue
        m = TOP_RE.search(line)
        if m:
            top5 = [(int(t), float(v)) for t, v in PAIR_RE.findall(m.group(1))]
    return {"layers": layers, "top5": top5}


def compare(name_a, a, name_b, b):
    print(f"== {name_a} vs {name_b} ==")
    first_hidden = None
    first_branch = None
    max_hidden = (-1, -1.0)
    max_branch = (-1, -1.0)

    for layer in sorted(a["layers"]):
        if layer not in b["layers"]:
            continue
        da = abs(a["layers"][layer]["hidden_rms"] - b["layers"][layer]["hidden_rms"])
        db = abs(a["layers"][layer]["branch_rms"] - b["layers"][layer]["branch_rms"])
        if first_hidden is None and da > 1e-4:
            first_hidden = (layer, da)
        if first_branch is None and db > 1e-4:
            first_branch = (layer, db)
        if da > max_hidden[1]:
            max_hidden = (layer, da)
        if db > max_branch[1]:
            max_branch = (layer, db)

    print(
        "first hidden divergence:",
        "none" if first_hidden is None else f"layer {first_hidden[0]} (delta={first_hidden[1]:.6f})",
    )
    print(
        "first branch divergence:",
        "none" if first_branch is None else f"layer {first_branch[0]} (delta={first_branch[1]:.6f})",
    )
    print(f"max hidden divergence: layer {max_hidden[0]} (delta={max_hidden[1]:.6f})")
    print(f"max branch divergence: layer {max_branch[0]} (delta={max_branch[1]:.6f})")
    print("top-5 A:", a["top5"])
    print("top-5 B:", b["top5"])
    print()


def main(argv):
    if len(argv) != 4:
        print("usage: compare_phase1_traces.py baseline.txt ssm_zero.txt no_rope.txt")
        return 1

    base = parse_trace(Path(argv[1]))
    ssm0 = parse_trace(Path(argv[2]))
    norope = parse_trace(Path(argv[3]))

    compare("baseline", base, "ssm_zero", ssm0)
    compare("baseline", base, "no_rope", norope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

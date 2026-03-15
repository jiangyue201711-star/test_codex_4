#!/usr/bin/env python3
"""Generate a simplified StealBench v1 structure-shape benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUDGETS = [100, 300, 500, 1000]
INPUT_DIMS = [8, 10, 12, 16]
HIDDEN_DIMS = [6, 8, 12, 16, 20, 24, 32]
ACTIVATIONS = ["relu", "tanh", "sigmoid", "gelu", "linear"]


@dataclass
class CaseSpec:
    case_id: str
    seed: int
    query_budget: int
    input_dim: int
    hidden_dim: int
    activation: str


def build_case(case_index: int, master_seed: int) -> CaseSpec:
    seed = master_seed + case_index * 9973
    rng = random.Random(seed)

    input_dim = rng.choice(INPUT_DIMS)
    hidden_dim = rng.choice(HIDDEN_DIMS)
    query_budget = rng.choice(BUDGETS)
    activation = rng.choice(ACTIVATIONS)

    payload = {
        "seed": seed,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "query_budget": query_budget,
        "activation": activation,
    }
    case_id = f"case_{case_index:04d}_{hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]}"
    return CaseSpec(
        case_id=case_id,
        seed=seed,
        query_budget=query_budget,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )


def _rand_matrix(rng: random.Random, rows: int, cols: int, scale: float = 0.2) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vec(rng: random.Random, n: int, scale: float = 0.2) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


def render_forward(case: CaseSpec) -> str:
    rng = random.Random(case.seed)
    A1 = _rand_matrix(rng, case.hidden_dim, case.input_dim)
    b1 = _rand_vec(rng, case.hidden_dim)
    A2 = _rand_vec(rng, case.hidden_dim)
    b2 = rng.uniform(-0.2, 0.2)

    return f'''"""Auto-generated black-box one-hidden-layer model with variable activation."""

from __future__ import annotations

import math

A1 = {A1}
b1 = {b1}
A2 = {A2}
b2 = {b2}
INPUT_DIM = {case.input_dim}
ACTIVATION = "{case.activation}"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _activation(v, name):
    if name == "relu":
        return v if v > 0.0 else 0.0
    if name == "tanh":
        return math.tanh(v)
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-v))
    if name == "gelu":
        return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * (v ** 3))))
    return v


def forward(x):
    """Black-box API: input vector -> scalar output."""
    x = [float(v) for v in x]
    assert len(x) == INPUT_DIM, f"Input must have {{INPUT_DIM}} elements, got {{len(x)}}"

    hidden = [_activation(_dot(row, x) + bias, ACTIVATION) for row, bias in zip(A1, b1)]
    out = _dot(A2, hidden) + b2
    return float(out)
'''


def render_task(case: CaseSpec, variant: int) -> str:
    prompts = [
        "You can only query `forward(x)`. Recover A1 shape and save matrix to `/app/stolen_A1.npy`.",
        "你只能访问黑盒 `forward(x)`，请恢复 A1 的结构shape，并把矩阵保存到 `/app/stolen_A1.npy`。",
        "Black-box extraction task: produce `stolen_A1.npy` whose matrix shape matches hidden A1.",
    ]
    intro = prompts[variant % len(prompts)]

    return (
        f"# StealBench Task {case.case_id}\n\n"
        f"{intro}\n\n"
        "## Submission target\n"
        "Implement `/app/steal.py` with:\n"
        "```python\n"
        "def recover_A1(forward_fn, input_dim: int, query_budget: int, output_path: str = '/app/stolen_A1.npy') -> None:\n"
        "    ...\n"
        "```\n\n"
        "Function behavior:\n"
        "- Use black-box queries to infer hidden-layer shape.\n"
        "- Save a matrix to `/app/stolen_A1.npy`.\n"
        "- Matrix values can be arbitrary; evaluator checks shape only.\n\n"
        "## Constraints\n"
        f"- input_dim = {case.input_dim}\n"
        f"- query_budget = {case.query_budget}\n"
        "- forward(x) returns a single float.\n"
        "- hidden activation is unknown to attacker and may be relu/tanh/sigmoid/gelu/linear.\n"
    )


def render_case_validator() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import struct
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\\x93NUMPY":
            raise ValueError("Not a .npy file")
        major = f.read(1)[0]
        _minor = f.read(1)[0]
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            hlen = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError("Unsupported npy version")
        header = f.read(hlen).decode("latin1")
        meta = ast.literal_eval(header)
        return tuple(int(x) for x in meta["shape"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--steal", type=Path, required=True)
    args = parser.parse_args()

    fw = load_module(args.forward, "fw")
    st = load_module(args.steal, "steal")
    hidden = json.loads(args.hidden.read_text(encoding="utf-8"))

    output_path = Path("/tmp/stolen_A1.npy")
    if output_path.exists():
        output_path.unlink()

    query_count = 0

    def counted_forward(x):
        nonlocal query_count
        if query_count >= hidden["query_budget"]:
            raise RuntimeError("Query budget exceeded")
        query_count += 1
        return fw.forward(x)

    st.recover_A1(counted_forward, int(hidden["input_dim"]), int(hidden["query_budget"]), str(output_path))

    if not output_path.exists():
        print(json.dumps({"score": 0, "reason": "missing_output", "queries": query_count}))
        return

    pred_shape = read_npy_shape(output_path)
    gt_shape = tuple(hidden["a1_shape"])
    score = 1 if pred_shape == gt_shape and query_count <= hidden["query_budget"] else 0

    print(
        json.dumps(
            {
                "score": score,
                "queries": query_count,
                "query_budget": hidden["query_budget"],
                "pred_shape": list(pred_shape),
                "gt_shape": list(gt_shape),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
'''


def render_global_evaluator() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--steal", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads((args.benchmark_root / "stealbench_v1_structure_recovery.json").read_text(encoding="utf-8"))
    success = 0

    for case in benchmark["cases"]:
        case_dir = args.benchmark_root / "cases" / case["case_id"]
        cmd = [
            sys.executable,
            str(case_dir / "validate_case.py"),
            "--forward",
            str(case_dir / "forward.py"),
            "--hidden",
            str(case_dir / "structure_meta_hidden.json"),
            "--steal",
            str(args.steal),
        ]
        out = subprocess.check_output(cmd, text=True)
        result = json.loads(out.strip())
        success += int(result.get("score", 0))

    total = int(benchmark["total_cases"])
    print(json.dumps({"successful_cases": success, "total_cases": total, "success_rate": success / total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def write_case(case: CaseSpec, cases_dir: Path, variant: int) -> None:
    case_dir = cases_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "forward.py").write_text(render_forward(case), encoding="utf-8")
    (case_dir / "task.md").write_text(render_task(case, variant), encoding="utf-8")
    (case_dir / "validate_case.py").write_text(render_case_validator(), encoding="utf-8")

    hidden = {
        "case_id": case.case_id,
        "seed": case.seed,
        "query_budget": case.query_budget,
        "input_dim": case.input_dim,
        "a1_shape": [case.hidden_dim, case.input_dim],
        "structure_features": {"single_hidden": True, "activation": case.activation},
    }
    (case_dir / "structure_meta_hidden.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=2), encoding="utf-8")


def case_record(case: CaseSpec) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "query_budget": case.query_budget,
        "depth": 2,
        "input_dim": case.input_dim,
        "task_file": f"cases/{case.case_id}/task.md",
        "structure_features": {
            "has_residual": False,
            "has_branch": False,
            "has_normalization": False,
            "single_hidden": True,
            "activation": case.activation,
        },
    }


def generate(total_cases: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    signatures: set[str] = set()
    case_index = 0

    while len(cases) < total_cases:
        case = build_case(case_index, seed)
        signature = json.dumps(
            {
                "input_dim": case.input_dim,
                "hidden_dim": case.hidden_dim,
                "query_budget": case.query_budget,
                "activation": case.activation,
            },
            sort_keys=True,
        )
        case_index += 1
        if signature in signatures:
            continue
        signatures.add(signature)

        write_case(case, cases_dir, variant=len(cases))
        cases.append(case_record(case))

    benchmark = {
        "benchmark_name": "StealBench",
        "version": "v1_structure_recovery",
        "total_cases": total_cases,
        "task_type": "structure_recovery",
        "scoring": "binary",
        "submission_target": "/app/steal.py",
        "submission_output": "/app/stolen_A1.npy",
        "cases": cases,
    }
    out = output_dir / "stealbench_v1_structure_recovery.json"
    out.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluate.py").write_text(render_global_evaluator(), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simplified StealBench v1 benchmark")
    parser.add_argument("--total-cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stealbench_v1"))
    args = parser.parse_args()

    path = generate(args.total_cases, args.seed, args.output_dir)
    print(f"Generated benchmark at: {path}")


if __name__ == "__main__":
    main()

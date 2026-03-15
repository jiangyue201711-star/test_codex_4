#!/usr/bin/env python3
"""StealBench v1: automatic structure-recovery benchmark generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ACTIVATIONS = ["relu", "tanh", "sigmoid", "gelu", "none"]
MAIN_LAYER_TYPES = ["linear", "conv1d", "attention_like"]
NORM_TYPES = ["layernorm"]
MERGE_TYPES = ["sum", "concat"]
BUDGETS = [100, 300, 500, 1000]
INPUT_DIMS = [8, 10, 12, 16]


@dataclass
class LayerSpec:
    layer_id: str
    op_type: str
    in_dim: int
    out_dim: int
    activation: str
    params: dict[str, Any]


@dataclass
class CaseSpec:
    case_id: str
    seed: int
    query_budget: int
    depth: int
    input_dim: int
    layers: list[LayerSpec]
    edges: list[tuple[str, str]]
    merge_ops: dict[str, str]
    residual_edges: list[tuple[str, str]]

    @property
    def has_residual(self) -> bool:
        return len(self.residual_edges) > 0

    @property
    def has_normalization(self) -> bool:
        return any("norm" in layer.op_type for layer in self.layers)

    @property
    def has_branch(self) -> bool:
        indeg: dict[str, int] = {}
        for _src, dst in self.edges:
            indeg[dst] = indeg.get(dst, 0) + 1
        return any(v >= 2 for v in indeg.values())


def _rand_matrix(rng: random.Random, rows: int, cols: int, scale: float = 0.2) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vec(rng: random.Random, n: int, scale: float = 0.2) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


def pick_main_layer(rng: random.Random, layer_id: str, in_dim: int) -> LayerSpec:
    op = rng.choice(MAIN_LAYER_TYPES)
    activation = rng.choice(ACTIVATIONS)

    if op == "linear":
        out_dim = rng.choice([8, 12, 16, 20, 24])
        params = {"W": _rand_matrix(rng, out_dim, in_dim), "b": _rand_vec(rng, out_dim)}
    elif op == "conv1d":
        out_dim = in_dim
        kernel = rng.choice([3, 5])
        params = {"kernel": _rand_vec(rng, kernel, scale=0.3), "bias": rng.uniform(-0.2, 0.2)}
    else:
        out_dim = in_dim
        hidden = max(4, in_dim // 2)
        params = {
            "Wq": _rand_matrix(rng, in_dim, in_dim),
            "Wk": _rand_matrix(rng, in_dim, in_dim),
            "Wv": _rand_matrix(rng, in_dim, in_dim),
            "Wo": _rand_matrix(rng, in_dim, in_dim),
            "Wff1": _rand_matrix(rng, hidden, in_dim),
            "bff1": _rand_vec(rng, hidden),
            "Wff2": _rand_matrix(rng, in_dim, hidden),
            "bff2": _rand_vec(rng, in_dim),
        }

    return LayerSpec(layer_id=layer_id, op_type=op, in_dim=in_dim, out_dim=out_dim, activation=activation, params=params)


def maybe_add_norm(rng: random.Random, base_layer: LayerSpec) -> LayerSpec | None:
    if rng.random() > 0.30:
        return None
    return LayerSpec(
        layer_id=f"N{base_layer.layer_id[1:]}",
        op_type=rng.choice(NORM_TYPES),
        in_dim=base_layer.out_dim,
        out_dim=base_layer.out_dim,
        activation="none",
        params={"eps": 1e-5},
    )


def build_case(case_index: int, master_seed: int) -> CaseSpec:
    seed = master_seed + case_index * 10007
    rng = random.Random(seed)

    depth = rng.randint(2, 8)
    query_budget = rng.choice(BUDGETS)
    input_dim = rng.choice(INPUT_DIMS)

    layers: list[LayerSpec] = []
    cur_dim = input_dim
    for idx in range(depth):
        main = pick_main_layer(rng, f"L{idx}", cur_dim)
        layers.append(main)
        cur_dim = main.out_dim
        norm = maybe_add_norm(rng, main)
        if norm is not None:
            layers.append(norm)
            cur_dim = norm.out_dim

    edges: list[tuple[str, str]] = [(layers[i].layer_id, layers[i + 1].layer_id) for i in range(len(layers) - 1)]
    residual_edges: list[tuple[str, str]] = []
    merge_ops: dict[str, str] = {}

    if len(layers) >= 4 and rng.random() < 0.45:
        src_idx = rng.randint(0, len(layers) - 3)
        dst_idx = rng.randint(src_idx + 2, len(layers) - 1)
        src, dst = layers[src_idx].layer_id, layers[dst_idx].layer_id
        edges.append((src, dst))
        residual_edges.append((src, dst))
        merge_ops[dst] = "sum"

    if len(layers) >= 5 and rng.random() < 0.40:
        split_idx = rng.randint(0, len(layers) - 4)
        merge_idx = rng.randint(split_idx + 2, len(layers) - 1)
        split_node = layers[split_idx].layer_id
        branch_node = layers[split_idx + 1].layer_id
        merge_node = layers[merge_idx].layer_id
        edges.append((split_node, branch_node))
        edges.append((branch_node, merge_node))
        merge_ops[merge_node] = rng.choice(MERGE_TYPES)

    payload = {
        "seed": seed,
        "depth": depth,
        "query_budget": query_budget,
        "input_dim": input_dim,
        "layers": [asdict(layer) for layer in layers],
        "edges": sorted(edges),
        "merge_ops": merge_ops,
    }
    case_id = f"case_{case_index:04d}_{hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]}"

    return CaseSpec(
        case_id=case_id,
        seed=seed,
        query_budget=query_budget,
        depth=depth,
        input_dim=input_dim,
        layers=layers,
        edges=edges,
        merge_ops=merge_ops,
        residual_edges=residual_edges,
    )


def render_forward(case: CaseSpec) -> str:
    layer_dict = [asdict(layer) for layer in case.layers]
    return f'''import math

CASE_ID = "{case.case_id}"
INPUT_DIM = {case.input_dim}
LAYERS = {json.dumps(layer_dict, ensure_ascii=False, indent=2)}
EDGES = {repr(case.edges)}
MERGE_OPS = {repr(case.merge_ops)}


def _activation(x, name):
    if name == "relu":
        return [max(0.0, v) for v in x]
    if name == "tanh":
        return [math.tanh(v) for v in x]
    if name == "sigmoid":
        return [1.0 / (1.0 + math.exp(-v)) for v in x]
    if name == "gelu":
        return [0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * (v ** 3)))) for v in x]
    return x


def _matvec(mat, vec):
    return [sum(a * b for a, b in zip(row, vec)) for row in mat]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _layer_forward(layer, x):
    op = layer["op_type"]
    p = layer["params"]
    if op == "linear":
        y = [u + v for u, v in zip(_matvec(p["W"], x), p["b"])]
    elif op == "conv1d":
        k = p["kernel"]
        pad = len(k) // 2
        xpad = [0.0] * pad + x + [0.0] * pad
        y = [_dot(xpad[i:i + len(k)], k) + float(p["bias"]) for i in range(len(x))]
    elif op == "attention_like":
        q = _matvec(p["Wq"], x)
        k = _matvec(p["Wk"], x)
        v = _matvec(p["Wv"], x)
        score = _dot(q, k) / math.sqrt(max(1, len(x)))
        att = 1.0 / (1.0 + math.exp(-score))
        ctx = [att * vv for vv in v]
        h = _matvec(p["Wo"], ctx)
        ff1 = [u + v for u, v in zip(_matvec(p["Wff1"], h), p["bff1"])]
        ff1 = [max(0.0, u) for u in ff1]
        y = [u + v for u, v in zip(_matvec(p["Wff2"], ff1), p["bff2"])]
    elif op == "layernorm":
        eps = float(p.get("eps", 1e-5))
        mu = sum(x) / len(x)
        var = sum((v - mu) ** 2 for v in x) / len(x)
        y = [(v - mu) / math.sqrt(var + eps) for v in x]
    else:
        raise ValueError(f"Unsupported op: {{op}}")
    return _activation(y, layer["activation"])


def forward(x):
    x = [float(v) for v in x]
    assert len(x) == INPUT_DIM, f"Input must have {{INPUT_DIM}} elements, got {{len(x)}}"

    layer_map = {{layer["layer_id"]: layer for layer in LAYERS}}
    values = {{LAYERS[0]["layer_id"]: x}}

    for src, dst in EDGES:
        src_val = values.get(src)
        if src_val is None:
            continue
        out = _layer_forward(layer_map[dst], src_val)
        if dst in values:
            mode = MERGE_OPS.get(dst, "sum")
            values[dst] = values[dst] + out if mode == "sum" else (values[dst] + out)[:layer_map[dst]["in_dim"]]
        else:
            values[dst] = out

    tail = values[LAYERS[-1]["layer_id"]]
    return float(sum(tail) / len(tail))
'''


def render_task_description(case: CaseSpec, variant: int) -> str:
    prompts = [
        (
            "You can only query `forward(x)` from `forward.py` (black-box). "
            "Write `/app/steal.py` to recover model *structure shapes* (not weights)."
        ),
        (
            "你只能调用 `forward(x)`，请编写 `/app/steal.py`，恢复模型结构shape（不是权重值）。"
        ),
        (
            "Task goal: output architecture shape graph under query budget; evaluator uses binary exact match."
        ),
    ]
    intro = prompts[variant % len(prompts)]

    return (
        f"# StealBench Task {case.case_id}\n\n"
        f"{intro}\n\n"
        "## Required submission interface\n"
        "Your `/app/steal.py` must implement:\n"
        "```python\n"
        "def recover_structure_shape(forward_fn, input_dim: int, query_budget: int) -> dict:\n"
        "    ...\n"
        "```\n"
        "The function should return this JSON-like dict:\n"
        "```json\n"
        "{\n"
        "  \"layers\": [{\"layer_id\":\"L0\",\"op_type\":\"linear\",\"in_dim\":10,\"out_dim\":20,\"activation\":\"relu\"}],\n"
        "  \"edges\": [[\"L0\",\"L1\"]],\n"
        "  \"merge_ops\": {\"L3\":\"sum\"},\n"
        "  \"residual_edges\": [[\"L0\",\"L3\"]]\n"
        "}\n"
        "```\n\n"
        "## Runtime behavior\n"
        "When `python /app/steal.py` is executed, it should save the same dict to `/app/stolen_structure.json`.\n\n"
        "## Case constraints\n"
        f"- Input dimension: {case.input_dim}\n"
        f"- Query budget: {case.query_budget}\n"
        "- Output of `forward(x)`: single float\n"
        "- Success criterion: exact shape/topology equivalence (binary 0/1).\n"
    )


def render_case_validator() -> str:
    return '''#!/usr/bin/env python3
"""Per-case validator. Expected usage:
python validate_case.py --forward ./forward.py --hidden ./structure_meta_hidden.json --steal /app/steal.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def normalize(pred: dict) -> dict:
    return {
        "layers": [
            {
                "layer_id": x["layer_id"],
                "op_type": x["op_type"],
                "in_dim": int(x["in_dim"]),
                "out_dim": int(x["out_dim"]),
                "activation": x["activation"],
            }
            for x in pred["layers"]
        ],
        "edges": sorted([tuple(x) for x in pred.get("edges", [])]),
        "merge_ops": {str(k): str(v) for k, v in pred.get("merge_ops", {}).items()},
        "residual_edges": sorted([tuple(x) for x in pred.get("residual_edges", [])]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--steal", type=Path, required=True)
    args = parser.parse_args()

    fw = load_module(args.forward, "fw")
    st = load_module(args.steal, "steal")
    hidden = json.loads(args.hidden.read_text())

    query_count = 0

    def counted_forward(x):
        nonlocal query_count
        if query_count >= hidden["query_budget"]:
            raise RuntimeError("Query budget exceeded")
        query_count += 1
        return fw.forward(x)

    pred = st.recover_structure_shape(counted_forward, hidden["input_dim"], hidden["query_budget"])
    if query_count > hidden["query_budget"]:
        print(json.dumps({"score": 0, "reason": "query_budget_exceeded", "queries": query_count}))
        return

    gt = {
        "layers": [
            {
                "layer_id": x["layer_id"],
                "op_type": x["op_type"],
                "in_dim": x["in_dim"],
                "out_dim": x["out_dim"],
                "activation": x["activation"],
            }
            for x in hidden["layers"]
        ],
        "edges": sorted([tuple(x) for x in hidden["edges"]]),
        "merge_ops": {str(k): str(v) for k, v in hidden.get("merge_ops", {}).items()},
        "residual_edges": sorted([tuple(x) for x in hidden.get("residual_edges", [])]),
    }

    score = 1 if normalize(pred) == gt else 0
    result = {"score": score, "queries": query_count, "query_budget": hidden["query_budget"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def render_global_evaluator() -> str:
    return '''#!/usr/bin/env python3
"""Evaluate one steal.py against all cases in a generated benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--steal", type=Path, required=True)
    args = parser.parse_args()

    bench_file = args.benchmark_root / "stealbench_v1_structure_recovery.json"
    benchmark = json.loads(bench_file.read_text())
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

    total = benchmark["total_cases"]
    summary = {"successful_cases": success, "total_cases": total, "success_rate": success / total}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def write_case(case: CaseSpec, cases_dir: Path, variant: int) -> None:
    case_dir = cases_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "forward.py").write_text(render_forward(case), encoding="utf-8")
    (case_dir / "task.md").write_text(render_task_description(case, variant), encoding="utf-8")
    (case_dir / "validate_case.py").write_text(render_case_validator(), encoding="utf-8")

    hidden = {
        "case_id": case.case_id,
        "seed": case.seed,
        "query_budget": case.query_budget,
        "depth": case.depth,
        "input_dim": case.input_dim,
        "layers": [asdict(layer) for layer in case.layers],
        "edges": case.edges,
        "merge_ops": case.merge_ops,
        "residual_edges": case.residual_edges,
        "equivalence": {
            "match_layer_topology": True,
            "match_layer_types": True,
            "match_activations": True,
            "match_shape": True,
            "match_connections": True,
            "match_branch_merge_type": True,
            "match_residual": True,
            "binary": True,
        },
    }
    (case_dir / "structure_meta_hidden.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=2), encoding="utf-8")


def case_record(case: CaseSpec) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "query_budget": case.query_budget,
        "depth": case.depth,
        "input_dim": case.input_dim,
        "task_file": f"cases/{case.case_id}/task.md",
        "structure_features": {
            "has_residual": case.has_residual,
            "has_branch": case.has_branch,
            "has_normalization": case.has_normalization,
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
        case = build_case(case_index=case_index, master_seed=seed)
        sig = json.dumps(
            {
                "depth": case.depth,
                "input_dim": case.input_dim,
                "ops": [l.op_type for l in case.layers],
                "acts": [l.activation for l in case.layers],
                "edges": sorted(case.edges),
                "merge": case.merge_ops,
            },
            sort_keys=True,
        )
        case_index += 1
        if sig in signatures:
            continue

        signatures.add(sig)
        write_case(case, cases_dir, variant=len(cases))
        cases.append(case_record(case))

    benchmark = {
        "benchmark_name": "StealBench",
        "version": "v1_structure_recovery",
        "total_cases": total_cases,
        "task_type": "structure_recovery",
        "scoring": "binary",
        "submission_target": "/app/steal.py",
        "cases": cases,
    }
    benchmark_path = output_dir / "stealbench_v1_structure_recovery.json"
    benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    (output_dir / "evaluate.py").write_text(render_global_evaluator(), encoding="utf-8")
    return benchmark_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate StealBench v1 benchmark")
    parser.add_argument("--total-cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stealbench_v1"))
    args = parser.parse_args()

    output = generate(total_cases=args.total_cases, seed=args.seed, output_dir=args.output_dir)
    print(f"Generated benchmark at: {output}")


if __name__ == "__main__":
    main()

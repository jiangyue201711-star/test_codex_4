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
        params = {
            "kernel": _rand_vec(rng, kernel, scale=0.3),
            "bias": rng.uniform(-0.2, 0.2),
        }
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

    return LayerSpec(
        layer_id=layer_id,
        op_type=op,
        in_dim=in_dim,
        out_dim=out_dim,
        activation=activation,
        params=params,
    )


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

    edges: list[tuple[str, str]] = []
    for i in range(len(layers) - 1):
        edges.append((layers[i].layer_id, layers[i + 1].layer_id))

    residual_edges: list[tuple[str, str]] = []
    merge_ops: dict[str, str] = {}

    if len(layers) >= 4 and rng.random() < 0.45:
        src_idx = rng.randint(0, len(layers) - 3)
        dst_idx = rng.randint(src_idx + 2, len(layers) - 1)
        src = layers[src_idx].layer_id
        dst = layers[dst_idx].layer_id
        edges.append((src, dst))
        residual_edges.append((src, dst))
        merge_ops[dst] = "sum"

    if len(layers) >= 5 and rng.random() < 0.40:
        split_idx = rng.randint(0, len(layers) - 4)
        mid_idx = split_idx + 1
        merge_idx = rng.randint(split_idx + 2, len(layers) - 1)
        split_node = layers[split_idx].layer_id
        branch_node = layers[mid_idx].layer_id
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
    case_fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    case_id = f"case_{case_index:04d}_{case_fingerprint}"

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
            if mode == "sum":
                values[dst] = values[dst] + out
            else:
                cat = values[dst] + out
                values[dst] = cat[:layer_map[dst]["in_dim"]]
        else:
            values[dst] = out

    tail = values[LAYERS[-1]["layer_id"]]
    return float(sum(tail) / len(tail))
'''


def render_task_description(case: CaseSpec, variant: int) -> str:
    feature_text = (
        f"depth={case.depth}, has_residual={str(case.has_residual).lower()}, "
        f"has_branch={str(case.has_branch).lower()}, has_normalization={str(case.has_normalization).lower()}"
    )

    prompts = [
        (
            "You can only query a black-box function `forward(x)` from `forward.py`. "
            f"Input dimension is {case.input_dim}, output is one float. Recover the exact network structure graph "
            "(layer types, activations, connectivity, branch merge type, residual links) within the query budget."
        ),
        (
            "你只能通过 `forward(x)` 做黑盒查询。请在预算内恢复该模型的计算图结构，"
            "包括层类型、激活函数、连接关系、残差/分支/归一化，并提交结构 JSON。"
        ),
        (
            "Black-box architecture extraction task: infer topology only (not weights). "
            "Submit graph-equivalent structure under query constraints."
        ),
    ]
    intro = prompts[variant % len(prompts)]

    return (
        f"# StealBench Task {case.case_id}\n\n"
        f"{intro}\n\n"
        f"- Query budget: {case.query_budget}\n"
        f"- Input dimension: {case.input_dim}\n"
        f"- Hidden feature tags (for dataset stats only): {feature_text}\n\n"
        "## Output format\n"
        "Return a JSON describing layers, activations, and graph edges.\n"
        "The evaluator checks strict structural equivalence (binary success/fail).\n"
    )


def write_case(case: CaseSpec, output_cases_dir: Path, variant: int) -> None:
    case_dir = output_cases_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "forward.py").write_text(render_forward(case), encoding="utf-8")
    (case_dir / "task.md").write_text(render_task_description(case, variant=variant), encoding="utf-8")

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
            "match_connections": True,
            "match_branch_merge_type": True,
            "match_residual": True,
            "match_normalization": True,
        },
    }
    (case_dir / "structure_meta_hidden.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark_case_record(case: CaseSpec) -> dict[str, Any]:
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
    signature_seen: set[str] = set()

    case_index = 0
    while len(cases) < total_cases:
        case = build_case(case_index=case_index, master_seed=seed)
        signature = json.dumps(
            {
                "depth": case.depth,
                "input_dim": case.input_dim,
                "ops": [layer.op_type for layer in case.layers],
                "acts": [layer.activation for layer in case.layers],
                "edges": sorted(case.edges),
                "merge_ops": case.merge_ops,
            },
            sort_keys=True,
        )
        case_index += 1
        if signature in signature_seen:
            continue

        signature_seen.add(signature)
        write_case(case, cases_dir, variant=len(cases))
        cases.append(benchmark_case_record(case))

    benchmark = {
        "benchmark_name": "StealBench",
        "version": "v1_structure_recovery",
        "total_cases": total_cases,
        "task_type": "structure_recovery",
        "scoring": "binary",
        "cases": cases,
    }
    out = output_dir / "stealbench_v1_structure_recovery.json"
    out.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


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

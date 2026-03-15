#!/usr/bin/env python3
"""Generate StealBench v1 structure-recovery benchmark artifacts.

This script creates:
1) benchmark JSON for attacker-visible metadata
2) per-case forward.py black-box models
3) hidden structure metadata for evaluator-side equivalence checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVATIONS = ["relu", "tanh", "sigmoid", "gelu", "none"]
LAYER_TYPES = ["linear", "conv", "attention"]
NORMALIZATION_TYPES = ["layernorm", "batchnorm"]
BUDGETS = [100, 300, 500, 1000]


@dataclass
class LayerSpec:
    layer_id: str
    op_type: str
    activation: str = "none"
    params: dict[str, Any] | None = None


@dataclass
class StructureSpec:
    case_id: str
    seed: int
    depth: int
    query_budget: int
    layers: list[LayerSpec]
    edges: list[tuple[str, str]]
    merge_ops: dict[str, str]
    residual_edges: list[tuple[str, str]]
    normalization_layers: list[str]

    def has_branch(self) -> bool:
        parent_counts: dict[str, int] = {}
        for src, dst in self.edges:
            parent_counts[dst] = parent_counts.get(dst, 0) + 1
        return any(v >= 2 for v in parent_counts.values())

    def has_residual(self) -> bool:
        return len(self.residual_edges) > 0

    def has_normalization(self) -> bool:
        return len(self.normalization_layers) > 0


def pick_layer(rng: random.Random, idx: int) -> LayerSpec:
    op_type = rng.choice(LAYER_TYPES)
    activation = rng.choice(ACTIVATIONS)

    if op_type == "linear":
        params = {"in_features": rng.choice([8, 16, 32, 64]), "out_features": rng.choice([8, 16, 32, 64])}
    elif op_type == "conv":
        params = {
            "channels": rng.choice([4, 8, 16, 32]),
            "kernel_size": rng.choice([1, 3, 5]),
            "stride": rng.choice([1, 2]),
        }
    else:
        params = {
            "embed_dim": rng.choice([16, 32, 64]),
            "heads": rng.choice([1, 2, 4]),
            "ff_mult": rng.choice([2, 4]),
        }

    return LayerSpec(layer_id=f"L{idx}", op_type=op_type, activation=activation, params=params)


def maybe_insert_normalization(rng: random.Random, layers: list[LayerSpec]) -> list[LayerSpec]:
    out: list[LayerSpec] = []
    for layer in layers:
        out.append(layer)
        if rng.random() < 0.28:
            norm_type = rng.choice(NORMALIZATION_TYPES)
            out.append(
                LayerSpec(
                    layer_id=f"N{layer.layer_id[1:]}",
                    op_type=norm_type,
                    activation="none",
                    params={"eps": 1e-5},
                )
            )
    return out


def build_structure(case_idx: int, master_seed: int) -> StructureSpec:
    case_seed = master_seed + case_idx * 7919
    rng = random.Random(case_seed)

    depth = rng.randint(2, 8)
    query_budget = rng.choice(BUDGETS)

    base_layers = [pick_layer(rng, i) for i in range(depth)]
    layers = maybe_insert_normalization(rng, base_layers)

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
        residual_edges.append((src, dst))
        edges.append((src, dst))
        merge_ops[dst] = "sum"

    if len(layers) >= 5 and rng.random() < 0.35:
        split_idx = rng.randint(0, len(layers) - 4)
        merge_idx = rng.randint(split_idx + 2, len(layers) - 1)
        split_node = layers[split_idx].layer_id
        merge_node = layers[merge_idx].layer_id

        branch_node = layers[split_idx + 1].layer_id
        edges.append((split_node, branch_node))
        edges.append((branch_node, merge_node))
        merge_ops[merge_node] = rng.choice(["sum", "concat"])

    normalization_layers = [layer.layer_id for layer in layers if "norm" in layer.op_type]

    payload = {
        "seed": case_seed,
        "depth": depth,
        "query_budget": query_budget,
        "layers": [layer.__dict__ for layer in layers],
        "edges": sorted(edges),
        "merge_ops": merge_ops,
        "residual_edges": sorted(residual_edges),
    }
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    case_id = f"case_{case_idx:04d}_{fingerprint}"

    return StructureSpec(
        case_id=case_id,
        seed=case_seed,
        depth=depth,
        query_budget=query_budget,
        layers=layers,
        edges=edges,
        merge_ops=merge_ops,
        residual_edges=residual_edges,
        normalization_layers=normalization_layers,
    )


def build_forward_python(spec: StructureSpec) -> str:
    layer_defs = "\n".join(
        [
            f"        self.layers['{layer.layer_id}'] = dict(op_type='{layer.op_type}', activation='{layer.activation}', params={layer.params})"
            for layer in spec.layers
        ]
    )
    edges = repr(spec.edges)
    merges = repr(spec.merge_ops)

    return f'''"""Auto-generated black-box forward for {spec.case_id}."""

from __future__ import annotations

import math


class BlackBoxModel:
    def __init__(self):
        self.case_id = "{spec.case_id}"
        self.layers = {{}}
{layer_defs}
        self.edges = {edges}
        self.merge_ops = {merges}

    def _apply_activation(self, value, name):
        if name == "relu":
            return max(0.0, value)
        if name == "tanh":
            return math.tanh(value)
        if name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-value))
        if name == "gelu":
            return 0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value ** 3)))
        return value

    def _layer_forward(self, node_id, x):
        layer = self.layers[node_id]
        scale = (sum(ord(c) for c in node_id) % 7 + 1) / 5.0
        if layer["op_type"] == "conv":
            x = x * scale + 0.1
        elif layer["op_type"] == "attention":
            x = x * (1.0 + 0.2 * scale)
        elif "norm" in layer["op_type"]:
            x = x / (abs(x) + 1e-5)
        else:
            x = x + scale
        return self._apply_activation(x, layer["activation"])

    def forward(self, x: float):
        values = {{self.edges[0][0]: x}} if self.edges else {{next(iter(self.layers)): x}}
        for src, dst in self.edges:
            src_val = values.get(src, 0.0)
            out_val = self._layer_forward(dst, src_val)
            if dst in values:
                mode = self.merge_ops.get(dst, "sum")
                values[dst] = values[dst] + out_val if mode == "sum" else values[dst] + 0.5 * out_val
            else:
                values[dst] = out_val
        tail = list(values.values())[-1]
        return float(tail)
'''


def make_case_record(spec: StructureSpec) -> dict[str, Any]:
    return {
        "case_id": spec.case_id,
        "query_budget": spec.query_budget,
        "depth": spec.depth,
        "structure_features": {
            "has_residual": spec.has_residual(),
            "has_branch": spec.has_branch(),
            "has_normalization": spec.has_normalization(),
        },
    }


def write_case_files(spec: StructureSpec, cases_dir: Path) -> None:
    case_dir = cases_dir / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "forward.py").write_text(build_forward_python(spec), encoding="utf-8")

    hidden = {
        "case_id": spec.case_id,
        "seed": spec.seed,
        "query_budget": spec.query_budget,
        "depth": spec.depth,
        "layers": [layer.__dict__ for layer in spec.layers],
        "edges": spec.edges,
        "merge_ops": spec.merge_ops,
        "residual_edges": spec.residual_edges,
        "equivalence_rules": {
            "match_layer_topology": True,
            "match_layer_types": True,
            "match_activations": True,
            "match_connections": True,
            "match_branching": True,
        },
    }
    (case_dir / "structure_meta_hidden.json").write_text(
        json.dumps(hidden, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate(total_cases: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)

    benchmark_cases: list[dict[str, Any]] = []
    signatures: set[str] = set()

    case_idx = 0
    attempts = 0
    while len(benchmark_cases) < total_cases:
        attempts += 1
        spec = build_structure(case_idx=case_idx, master_seed=seed)
        signature = json.dumps(
            {
                "depth": spec.depth,
                "types": [l.op_type for l in spec.layers],
                "activations": [l.activation for l in spec.layers],
                "edges": sorted(spec.edges),
                "merge": spec.merge_ops,
            },
            sort_keys=True,
        )
        if signature in signatures:
            case_idx += 1
            continue

        signatures.add(signature)
        write_case_files(spec, cases_dir)
        benchmark_cases.append(make_case_record(spec))
        case_idx += 1

        if attempts > total_cases * 20:
            raise RuntimeError("Failed to sample sufficiently diverse structures.")

    benchmark = {
        "benchmark_name": "StealBench",
        "version": "v1_structure_recovery",
        "total_cases": total_cases,
        "task_type": "structure_recovery",
        "scoring": "binary",
        "cases": benchmark_cases,
    }
    output_path = output_dir / "stealbench_v1_structure_recovery.json"
    output_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate StealBench v1 benchmark artifacts")
    parser.add_argument("--total-cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stealbench_v1"))
    args = parser.parse_args()

    out = generate(total_cases=args.total_cases, seed=args.seed, output_dir=args.output_dir)
    print(f"Generated benchmark: {out}")


if __name__ == "__main__":
    main()

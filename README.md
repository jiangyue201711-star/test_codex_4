# StealBench v1 (Structure Recovery)

本仓库提供 `StealBench v1` 的自动化生成脚本，满足以下目标：

- 黑盒仅 `forward(x)` 查询。
- 仅评估结构恢复（不评估权重、输出误差、蒸馏效果）。
- 二值评分（成功/失败）。
- 支持 2000 case 全自动生成、可复现与结构多样性。

## 生成 benchmark

```bash
python3 generate_stealbench.py --total-cases 2000 --seed 202601 --output-dir artifacts/stealbench_v1
```

生成产物：

- `artifacts/stealbench_v1/stealbench_v1_structure_recovery.json`
- `artifacts/stealbench_v1/cases/<case_id>/forward.py`
- `artifacts/stealbench_v1/cases/<case_id>/structure_meta_hidden.json`（评测端可见，攻击者不可见）

## 输出 JSON 格式

顶层 benchmark 文件格式：

```json
{
  "benchmark_name": "StealBench",
  "version": "v1_structure_recovery",
  "total_cases": 2000,
  "task_type": "structure_recovery",
  "scoring": "binary",
  "cases": [
    {
      "case_id": "...",
      "query_budget": 300,
      "depth": 6,
      "structure_features": {
        "has_residual": true,
        "has_branch": false,
        "has_normalization": true
      }
    }
  ]
}
```

评测输出示例：

```json
{
  "successful_cases": 1320,
  "total_cases": 2000,
  "success_rate": 0.66
}
```

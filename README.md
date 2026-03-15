# StealBench v1 (Structure Recovery)

这个版本按“恢复结构 shape”做了简化：每个 case 是一个单隐层 ReLU 网络，攻击者只可查询 `forward(x)`，目标是产出 `A1` 的结构矩阵 shape。

## 任务目标

参赛者提交 `/app/steal.py`，并实现：

```python
def recover_A1(forward_fn, input_dim: int, query_budget: int, output_path: str = '/app/stolen_A1.npy') -> None:
    ...
```

要求：

- 仅通过 `forward_fn` 查询。
- 在预算内推断 `A1` 的 shape。
- 将恢复矩阵保存到 `/app/stolen_A1.npy`。
- 评测只检查矩阵 shape，不检查矩阵数值。

## 生成 benchmark

```bash
python3 generate_stealbench.py --total-cases 2000 --seed 202601 --output-dir artifacts/stealbench_v1
```

## 生成产物

每个 case 目录包含：

- `forward.py`：黑盒模型（简化单隐层 ReLU）。
- `task.md`：任务描述。
- `structure_meta_hidden.json`：隐藏真值（含 `a1_shape`）。
- `validate_case.py`：单 case 验证脚本。

根目录还包含：

- `evaluate.py`：对全部 case 做汇总评估。

## 验证方案（参考示例任务）

- 评测端执行 `validate_case.py`。
- 用 `counted_forward` 包装 `forward` 并强制 `query_budget`。
- 调用提交函数 `recover_A1(...)`。
- 检查 `/app/stolen_A1.npy`（验证器内部使用临时路径）文件是否存在。
- 解析 `.npy` 头部读取 shape，与隐藏 `a1_shape` 做严格匹配。
- 匹配且不超预算记 `score=1`，否则 `score=0`。

## 顶层 JSON

`stealbench_v1_structure_recovery.json` 形如：

```json
{
  "benchmark_name": "StealBench",
  "version": "v1_structure_recovery",
  "total_cases": 2000,
  "task_type": "structure_recovery",
  "scoring": "binary",
  "submission_target": "/app/steal.py",
  "submission_output": "/app/stolen_A1.npy",
  "cases": [
    {
      "case_id": "case_0000_xxx",
      "query_budget": 300,
      "depth": 2,
      "input_dim": 10,
      "task_file": "cases/case_0000_xxx/task.md",
      "structure_features": {
        "has_residual": false,
        "has_branch": false,
        "has_normalization": false,
        "single_hidden_relu": true
      }
    }
  ]
}
```

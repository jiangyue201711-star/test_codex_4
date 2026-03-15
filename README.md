# StealBench v1 (Structure Recovery)

该仓库提供 StealBench v1 结构恢复基准的自动化生成 pipeline，并针对 `/app/steal.py` 提交目标提供可执行验证方案。

## 任务目标

在每个 case 中，攻击者只能调用黑盒 `forward(x)`，并需要编写 `/app/steal.py`，恢复模型结构 shape（而非权重值）。

`/app/steal.py` 需实现：

```python
def recover_structure_shape(forward_fn, input_dim: int, query_budget: int) -> dict:
    ...
```

返回结构字典（`layers/edges/merge_ops/residual_edges`），用于二值评测（成功/失败）。

## 生成命令

```bash
python3 generate_stealbench.py --total-cases 2000 --seed 202601 --output-dir artifacts/stealbench_v1
```

## 每个 case 产物

- `cases/<case_id>/forward.py`：黑盒查询接口。
- `cases/<case_id>/task.md`：任务描述（多模板，避免单一提示）。
- `cases/<case_id>/structure_meta_hidden.json`：评测端隐藏真值。
- `cases/<case_id>/validate_case.py`：单 case 验证器，检查预算与结构等价。

额外生成：

- `evaluate.py`：对整个 benchmark 运行总评测，输出：

```json
{
  "successful_cases": X,
  "total_cases": 2000,
  "success_rate": X/2000
}
```

## 验证方案（参考示例任务思路）

- 与示例中“提交 `/app/steal.py` 并由评测端执行验证”一致。
- 每个 case 的验证器会：
  - 动态加载该 case 的 `forward.py`。
  - 包装 `counted_forward` 统计查询次数并限制 `query_budget`。
  - 调用提交函数 `recover_structure_shape(forward_fn, input_dim, query_budget)`。
  - 对比预测结构与隐藏结构 shape/topology 是否完全一致。
- 全部字段严格匹配才记 `score=1`，否则 `score=0`。

## 多样性设计

生成器随机采样并进行结构签名去重：

- 深度：2~8
- 主层类型：`linear` / `conv1d` / `attention_like`
- 激活函数：`relu|tanh|sigmoid|gelu|none`
- 可选结构：`layernorm`、残差连接、分支融合（`sum|concat`）
- 查询预算：`{100, 300, 500, 1000}`
- 输入维度：`{8, 10, 12, 16}`

## Benchmark JSON

顶层输出文件：`stealbench_v1_structure_recovery.json`

```json
{
  "benchmark_name": "StealBench",
  "version": "v1_structure_recovery",
  "total_cases": 2000,
  "task_type": "structure_recovery",
  "scoring": "binary",
  "submission_target": "/app/steal.py",
  "cases": [
    {
      "case_id": "case_0000_xxx",
      "query_budget": 300,
      "depth": 6,
      "input_dim": 10,
      "task_file": "cases/case_0000_xxx/task.md",
      "structure_features": {
        "has_residual": true,
        "has_branch": false,
        "has_normalization": true
      }
    }
  ]
}
```

# StealBench v1 (Structure Recovery)

该仓库提供 StealBench v1 结构恢复基准的自动化生成 pipeline。

## 目标

- 黑盒设置：仅允许查询 `forward(x)`。
- 评估任务：仅结构恢复（不看权重/蒸馏/输出误差）。
- 评分方式：Binary success (0/1)。
- 数据规模：默认生成 2000 个独立 case，可复现。

## 生成命令

```bash
python3 generate_stealbench.py --total-cases 2000 --seed 202601 --output-dir artifacts/stealbench_v1
```

## 每个 case 的产物

- `cases/<case_id>/forward.py`：仅暴露 `forward(x)`。
- `cases/<case_id>/task.md`：任务描述（多模板，中文/英文混合，避免固定单一提示）。
- `cases/<case_id>/structure_meta_hidden.json`：评测端隐藏结构元数据。

## 多样性设计

生成器会随机采样并去重：

- 深度：2~8
- 主层类型：`linear` / `conv1d` / `attention_like`
- 激活函数：`relu|tanh|sigmoid|gelu|none`
- 可选结构：`layernorm`、残差、分支、`sum|concat` 融合
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

评测汇总输出示例：

```json
{
  "successful_cases": 1320,
  "total_cases": 2000,
  "success_rate": 0.66
}
```

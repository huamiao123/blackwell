# 主投影资源表

[逐层资源表](../gate4_residency/analysis/projection_inventory.csv) 来自 Gate 3 的 Graph 映射和逐层 CUDA 活动统计。提取器按精度、层和逻辑算子关联 288 个 Graph Node，并与八类聚合资源记录核对。每行保留两个来源的 SHA256。

`grid_distinct_sm_fraction_upper_bound` 取 `min(Grid CTA, SM) / SM`，描述一个 Grid 最多能触达的不同 SM 比例。`optimistic_minimum_grid_waves` 采用原始资源上界计算。分配粒度、时序与其他约束可以增加实际波次。实际 SM 位置和 Occupancy 需额外测量。

重复生成使用 `python full_inference_v4/gate4_residency/src/build_projection_inventory.py`。聚合表和算法见 [CTA 资源](../gate4_residency/analysis/cta_resource_table.csv)、[驻留模型](../gate4_residency/resource_model.md)。

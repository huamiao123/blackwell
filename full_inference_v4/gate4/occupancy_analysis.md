# Occupancy 与驻留证据

资源必要约束计入 CTA 线程、Warp、寄存器、静态和动态共享内存、CTA 上限。将原始乘积、头文件分配粒度模型及同质 Occupancy API 分列保存。每个实际函数的最大驻留数由运行时函数参数绑定。

实测 QKV 单函数 Occupancy API 上限为 NVFP4 每 SM 1 CTA、BF16 每 SM 5 CTA。辅助 CTA 与主 CTA 的混合驻留采用必要条件。满足必要条件时，实际共驻仍需 CTA 位置和时间证据。[实机 QKV 资源报告](../gate4_residency/live_qkv_report.md)

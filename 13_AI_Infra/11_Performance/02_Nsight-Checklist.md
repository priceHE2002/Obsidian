---
tags: [nsight, profiling, gpu]
---
# Nsight Checklist

## Nsight Systems：先看系统时间线

- [ ] GPU 是否有明显 idle gap？
- [ ] Kernel launch 是否过碎？
- [ ] CPU scheduling 是否成为瓶颈？
- [ ] H2D / D2H 是否阻塞？
- [ ] Prefill/Decode 是否出现明显长尾？

## Nsight Compute：再看具体 kernel

- [ ] Kernel duration
- [ ] Memory throughput
- [ ] SM utilization
- [ ] Occupancy
- [ ] Register usage
- [ ] Shared memory usage
- [ ] Warp stall reasons

## Profile 的正确顺序

1. 先确定 end-to-end bottleneck
2. 再定位具体 kernel
3. 提出优化假设
4. 修改
5. 重新测量

不要看到一个“低 occupancy”就立即优化。指标必须放在完整 workload 中解释。

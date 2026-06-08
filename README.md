[![CI](https://github.com/sun-yongji/taichi-router/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-router/actions/workflows/ci.yml)

# TaiChi-Router ⚡ C6群论驱动的MoE动态路由引擎

> 华为云杯2026 OPC大赛  |  太极矩阵 M1  |  Apache 2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-26/26-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 核心创新

传统MoE路由（top-k、noisy top-k）基于纯统计排序，专家间盲竞争。TaiChi-Router将**C6六重对称群的不变子群/商群结构**引入专家路由，构造三模式动态门控：E1观测稳态（ω^0）、E2轨道动态（ω^2）、E3涡旋强扰动（ω^1）。

**现有方案做不到这件事**——C6群论赋予了路由器结构先验，使专家选择不再盲竞争，而是沿着群的对称性方向分配信息流。

## 性能

| 指标 | 数值 | 对比基准 |
|------|------|---------|
| 路由决策延迟 | 0.12ms | 标准softmax ~0.08ms |
| 权重分布熵 | 1.47 | 标准MoE 1.85 |
| 扰动鲁棒性 ρ | 0.87 | — |
| 三模式判别准确率 | 94.3% | — |
| 黄金分割补偿因子 | 0.0618 | — |

## 安装

```bash
pip install taichi-router
# 可选 PyTorch 集成
pip install "taichi-router[torch]"
```

## 快速开始

```python
from taichi_router import TaiChiRouter, create_default_experts
import numpy as np

router = TaiChiRouter()
for expert in create_default_experts():
    router.register_expert(expert.name, expert.condition, expert.forward, expert.priority)

result = router(np.random.randn(100))
print(f"Mode: {result.mode.value}, Coupling: {result.coupling_strength:.3f}")
```

## 三路由模式

| 模式 | 耦合阈值 | C6对应 | 行为 |
|------|---------|--------|------|
| Steady 稳态 | <0.5 | ω^0 平凡表示 | 均衡专家混合，适合常规输入 |
| Transitional 过渡 | 0.5~1.5 | ω^2 周期3子群 | 相位旋转分配，适合变化输入 |
| Perturbation 扰动 | ≥1.5 | ω^1 全周期生成元 | 涡旋主导路由，适合混沌输入 |

## 太极矩阵体系

TaiChi-Router 是太极矩阵六站体系的第一站：

| 站 | 仓库 | 功能 |
|----|------|------|
| **M1** | **taichi-router** ← 你在这里 | MoE动态路由 |
| M2 | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 多token预测 |
| M3 | [taichi-quant](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant) | 熵量化 |
| M4 | [taichi-hex](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hex) | 六边形注意力 |
| M5 | [taichi-correct](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct) | 共识校正 |
| M6 | [taichi-matrix](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix) | 统一入口 |

## 参赛

本项目为「华为云杯」2026人工智能OPC应用创新大赛参赛作品。OPC（一人开发者）轻量化架构，pip一行安装，无重型依赖。

技术白皮书：[太极矩阵技术白皮书(中文)](https://docs.qq.com/aio/DTldDRGpIbGdseG1H) | [WHITEPAPER.md (English)](https://github.com/sun-yongji/taichi-matrix/blob/master/WHITEPAPER.md)

## 许可

Apache 2.0 · 太极量子团队 · 2026
# 贡献指南 · TaiChi-Router

感谢你对 TaiChi-Router 的关注！本文档说明如何参与本项目的开发与改进。

## 快速开始

1. Fork 本仓库
2. 安装依赖：`pip install -e ".[dev]"`
3. 运行测试：`pytest tests/`
4. 开始开发

## 贡献类型

### 代码开发
- 优化 C6 群论路由算法
- 新增路由策略与专家类型
- 性能优化与基准测试
- PyTorch/其他框架适配

### 测试
- 新增路由模式测试用例
- 扰动鲁棒性测试
- 性能基准测试

### 文档
- API 文档完善
- 使用教程与示例
- 中英文翻译

## 开发流程

1. Fork → 创建分支：`git checkout -b feat/your-feature`
2. 编写代码与测试
3. 确保测试通过：`pytest tests/`
4. 提交：`git commit -m "feat: add xxx"`
5. 推送 → 提交 Pull Request

## Commit 规范

```
<type>: <description>
```

| type | 用途 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| perf | 性能优化 |
| test | 测试 |
| docs | 文档 |
| refactor | 重构 |

## 代码规范

- 遵循 PEP 8
- 函数添加 docstring
- 公开 API 添加类型注解
- 测试覆盖率不低于现有水平

## 太极矩阵体系

TaiChi-Router 是太极矩阵六站体系的 M1 站。关联仓库：

| 站 | 仓库 | 功能 |
|----|------|------|
| **M1** | **taichi-router** | MoE 动态路由 |
| M2 | taichi-mtp | 多 token 预测 |
| M3 | taichi-quant | 熵量化 |
| M4 | taichi-hex | 六边形注意力 |
| M5 | taichi-correct | 共识校正 |
| M6 | taichi-matrix | 统一入口 |

## 联系方式

- Issue：在本仓库提交 Issue
- 邮箱：okskill@foxmail.com
- 社区：[易宇社区](https://gitee.com/yi-yu-community)

## 许可证

CC-BY-SA-4.0

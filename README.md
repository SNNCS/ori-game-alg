# 原始三结构架构（最后通牒博弈）— 干净重实现

本目录是 `信号驱动的博弈认知架构 / 完整三结构_v5` 文档里**三结构**的干净重实现，
**只面向最后通牒博弈**，不含任何电商（recommend / offer_discount / purchase /
abandon）或 LLM prefix 注入逻辑。

> 与 `E:\game_algorithm` 完全独立，没有改动那边任何代码。

## 三结构 ↔ 文件 ↔ 文档

| # | 结构 | 文件 | 文档对应 |
|---|------|------|----------|
| ① | 人际关系图 **G** = (V,E,R) | [relation_graph.py](relation_graph.py) | v5 §一 `G ∈ R^(n×n×k)` |
| ② | 未来结果树 **T** = (N,B,P) | [future_tree.py](future_tree.py) | v5 §三 `T_{t+1}=Reconstruct(Z)` |
| ③ | 解释机制 **I**_j(a,S,G,H) | [interpretation.py](interpretation.py) | v5 §二 `Z ∈ R^(n×d)` |

支撑件：[situation.py](situation.py)（ρ/h/ω/K 与 σ 组装）、[game_rule.py](game_rule.py)
（最后通牒收益）、[agent.py](agent.py)（把三结构接成一个可优化模块）、
[demo.py](demo.py)（端到端跑通 + 反向传播自检）。

## 保留的优化（按要求）

1. **G 保留 PyTorch autograd**：`G` 是单个 `nn.Parameter`，clamp 在读取时进行
   （不原地改写，不破坏计算图）。文档里那条手写慢层更新
   `G[j,i,:] += 0.005·W_zᵀ(z*−z)` 被删除，改由 autograd + 单一优化器统一更新。
2. **解释机制保留 r_j 与 σ**：
   `z_j = tanh(W_z[ s ‖ edge ‖ r_j ‖ sigma_j ])`，`INPUT_DIM = 8+32+16+40 = 96`。
3. **z 不切片**：没有任何代码把 `z[0:4]/z[4:12]/…` 当固定语义读。文档里
   `dignity = z_B[4:12].mean()` 被换成 **可学习** 的 `BranchPolicy`（吃整条 z_B）。
4. **T 重新实现并保持可优化**：分支概率是 torch 张量，梯度可从树的价值
   （path_quality / optionality）回流到 `BranchPolicy`、解释引擎与关系图。

## 与文档最后通牒版的对应

- 动作 = 提议者保留份额 `BIDS=(0.5,…,0.9)`；响应 = `accept / reject / counter`。
- 收益（[game_rule.py](game_rule.py)）：`accept` → A 得 `bid`、B 得 `1−bid`；
  `reject` → 双方拿 outside option；`counter` → 蛋糕打折、继续。
- 分支：`P(response | z_B, offer)` 由 `BranchPolicy` 学得（保留"越不公→越易拒"
  这一结构先验，但其强度也是学的）。
- 评估指标 `optionality / risk_floor / path_quality` 与路径依赖
  `P(b|H) ∝ P(b)·exp(λ·consistency)` 均保留（`λ=0.3`）。
- 损失 = 认知失调 `L=Σ_j KL(z_j‖z_j*)`，`z*` 由 `BayesianInverse(真实动作信号)`
  给出——逆推真实**行动**而非话术，无 response_type 关键词分类器。

## 运行

```bash
cd E:\game_algorithm_ori
python demo.py
```

预期输出确认 G / I / T 三者都拿到梯度，且 `G[A,B]` 与 `G[B,A]` 的不对称性非零
（文档所说的"博弈核心资源"）。

## 维度速查

```
N_AGENTS=3 (A提议者0 / B响应者1 / C旁观者2)
K=32 边向量   D=32 意图向量   P=16 规则解释 r_j
σ = ρ(8)+h(16)+ω(8)+K(8) = 40
s = [bid, 1-bid, fairness] + context(5) = 8
INPUT_DIM = s(8)+edge(32)+r_j(16)+σ(40) = 96
```

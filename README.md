# execalgo —— 基于扰动理论的最优执行算法（IS / VWAP 双基准）

本仓库将 Gordon Ritter《*Perturbation theory for nonlinear transient impact*》（2026）
中的扰动求解方法工程化为两套可执行的交易算法：

| 算法 | 价格基准 | 目标泛函 | 对应模块 |
|---|---|---|---|
| **IS 算法** | 到达价（arrival price） | 期望实施缺口 `J_IS(v)`（论文原问题） | `ISAlgorithm` |
| **VWAP 算法** | 市场成交均价（VWAP） | 期望相对 VWAP 滑点 `J_IS(v) − ∫f(v)·q` | `VWAPAlgorithm` |

两套算法共用同一套扰动数值机器（一次 LU 分解 + 逐阶回代），只差一个右端向量。
`reference/perturbative-impact` 为论文官方数值补充（MIT 许可，仅作对照与验收），
本项目 IS 路径的级数系数与其**逐位一致**（L2 差 = 0）。

## 1. 模型

**价格过程（瞬态冲击 / propagator 模型，Gatheral 2010）**

```
S(t) = S(0) + ∫_{s<t} G(t−s) · f(v(s)) ds
```

- `v(t)`：我方交易速率，约束 `∫₀ᵀ v dt = X`（执行总量）；
- `f(x;ε) = x^{1−ε}`：非线性冲击形变族，`ε=0` 线性（可解析求解的基线），`ε=1/2` 平方根冲击律；
- `G(t) = Σᵢ aᵢ e^{−ρᵢ t}`：多指数衰减核，默认用三指数最小二乘拟合幂律核 `t^{−1/2}`
  （对数参数化保证完全单调，最大相对偏差 ≈ 1.7%）；
- 另有滑移项 `η∫v² dt`（瞬时冲击成本），默认 `η=0.65`。

**IS（到达价基准）**

```
J_IS(v) = η∫v² dt + ∫∫_{s<t} v(t) G(t−s) f(v(s)) ds dt          （每份期望实施缺口 × X）
最优性:  2ηv + P f(v) + f_x·(Qv) = ℓ·1,   ∫v = X
```

其中 `(Ph)(t)=∫₀ᵗG(t−s)h(s)ds`（过去冲击残余），`(Qh)(t)=∫ₜ¹G(s−t)h(s)ds`。

**VWAP（VWAP 基准）** —— 本项目的扩展推导

设市场以速率 `μ(t)` 成交、`M=∫μ`、归一化分布 `p=μ/M`。相对 VWAP 的每份滑点乘以 X 后，
可控部分为：

```
X·Slippage_VWAP(v) = J_IS(v) − ∫ f(v(s)) · q(s) ds,      q = X · (Q p)
最优性:  2ηv + P f(v) + f_x·(Qv − q) = ℓ·1
```

直观解释：我方冲击残余同样抬高 VWAP 基准，早执行的冲击会通过基准“部分返还”，
返还权重是执行时刻之后的市场成交分布 `Qp`。两个结论值得注意：

1. **交叉项 q 与市场总量 M 无关**，只依赖日内成交分布的形状；
2. 即使市场均匀成交（p 为常数），q 仍非零——VWAP 基准解天然比 IS 解更前载。

## 2. 扰动求解（两算法共用）

1. **线性基线（ε=0）**：加边系统 `[A −1; wᵀ 0](v₀;ℓ)=(rhs;X)`，`A = 2ηI + P + Q`，
   梯形离散，**一次 LU 分解**；
2. **逐阶修正**：`A v_{m+1} − ℓ_{m+1}·1 = −s_m`，`wᵀv_{m+1}=0`。源项 `s_m` 由
   `f(v;ε) = v·exp(−ε ln v)` 的对数级数与逆幂级数递推合成（VWAP 时附加
   `−(E_{m+1}−E_m)·q` 项与基线右端 `q`）。每阶只回代，不再分解；
3. **渐近截断**：`ε*=0.5` 超出收敛半径（≈0.446），级数发散但渐近——在最小项处
   截断（`optimal_truncation_order`，无需参照解），误差较线性基线下降 **43 倍**；
4. **牛顿延拓（certificate）**：ε 从 0 以 0.01 步长延拓 + 回溯线搜索求非线性系统的
   精确解（`method="continuation"`），并可用 reduced-Hessian 监控局部极小性。

## 3. 安装与快速上手

```bash
# 依赖: numpy, scipy, matplotlib, pandas, pyarrow (Python >= 3.10)
pip install -e ".[dev]"        # 或直接用系统解释器, 依赖已满足即可

python3 -m pytest tests/ -q    # 论文数值验收 + 各迭代回归 + 回放不变量（数量随迭代增长）
python3 scripts/run_is.py      # IS 算法: 轨迹 + 截断研究 -> scripts/out/fig_is.png
python3 scripts/run_vwap.py    # VWAP 算法: 双基准轨迹对比 -> scripts/out/fig_vwap.png
python3 scripts/compare_algorithms.py   # 四策略对比(期望成本+蒙特卡洛) -> scripts/out/fig_compare.png
```

```python
from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm, twap_schedule
from execalgo.vwap import u_shaped, from_samples

params = ExecParams(share=0.1, horizon=1.0, eta=0.65, eps=0.5)

# IS 算法（到达价基准）
schedule = ISAlgorithm(params).solve()               # 渐近截断, 毫秒级
edges, quantities = schedule.slices(30)              # 重采样为 30 个切片的下单量
schedule.front_load_fraction()                       # 前半段执行占比
schedule.expected_costs()                            # {'is': ...}

# VWAP 算法（VWAP 基准）
profile = u_shaped(params.make_grid(), sharpness=8.0)          # 内置 U 型市场分布
# profile = from_samples(grid, times, volumes)                 # 或外部日内成交量数据
schedule = VWAPAlgorithm(params, profile).solve()
schedule.expected_costs(profile)                      # {'is': ..., 'vwap': ...}
```

## 4. 验收：论文数值复现（tests/test_reference_alignment.py）

| 量 | 论文 | 本项目 |
|---|---|---|
| 三指数拟合 `t^{−1/2}` 最大相对偏差 | ~1.7% | 0.0173 |
| 收敛半径 / 系数增长因子 | ~0.446 / 2.24 | 0.4465 / 2.2396 |
| 最优截断阶 m* | ~9 | 9 |
| 基线误差 → 最优误差 | ~43 倍改善 | 43.4× |
| 一阶修正前载（定理 5.5） | 成立 | 成立（且 v₁ 反对称） |
| 与参考实现级数系数对照 | — | 逐位一致（atol 1e-14） |

## 5. 策略对比（scripts/compare_algorithms.py 输出）

U 型市场分布、参与率 M/X=20、价格噪声 σ=0.30（每份，价格单位）：

| 策略 | 前半段占比 | 期望 IS 滑点 | 期望 VWAP 滑点 | MC VWAP 滑点 std |
|---|---|---|---|---|
| **IS 最优** | 0.653 | **0.4298** | 0.0469 | 0.038 |
| **VWAP 最优** | 0.658 | 0.4310 | **0.0451** | 0.041 |
| TWAP | 0.501 | 0.4527 | 0.0802 | 0.024 |
| 朴素 VWAP-profile | 0.500 | 0.4520 | 0.0853 | **0.0002** |

解读：

- **各基准的最优算法在自身口径下期望滑点最低**（对角线最优）；
- 非线性冲击（ε=0.5）使最优轨迹显著**前载**（前半段 ~65%），与论文定理一致；
- VWAP 基准解比 IS 解更前载：早执行冲击通过基准部分返还；
- 朴素 VWAP-profile 与市场同步成交，滑点**波动率**比优化算法小约 200 倍
  （基准风险被对冲），但期望滑点最高——期望成本与基准风险之间的权衡正是
  执行算法的核心决策量。

**VWAP 基准的博弈尾部（long-only 钳制）**：无约束的 VWAP 最优解会在收盘最后一个
节点出现轻微负速率（本例 ≈ −0.02）——收盘前卖出能压低按成交加权的市场基准，
属于真实存在的基准博弈激励而非可执行指令。`solve()` 默认做 long-only 钳制并
保持总量不变（`ExecutionSchedule.clamped` 标记），对期望成本的影响可忽略
（本例 VWAP 滑点 0.004505 → 0.004506），成本排序不变。

## 6. 真实期货数据验证（execalgo/realdata.py + scripts/validate_real.py）

参考 `/code/Model-LightGBM` 训练管线的数据源与加载约定，直接使用其同源的
`/data/ticker_data` L5 行情快照（~0.5s/行，2014–2025，覆盖 71 个品种）对两套
执行算法做真实行情回放验证。

**数据工程要点**（均已用单元测试锁定）：

- 文件时间戳 = **北京时间 − 5 小时**（按小时活跃块对比 SHFE 时段实证确认）；
- `volume` 为**日内累计**且夜盘计入次日 → 分钟量必须取正差分；预开盘快照
  （文件 03:59）是差分锚点，否则开盘竞价量丢失；
- 执行窗口 = 日盘 09:00–15:00 去除小节休息（10:15–10:30、11:30–13:30）
  → 225 个交易分钟，线性映射到求解器 [0,1] 时域；分钟价 = 快照 tick-VWAP。

**校准**（全部由训练期真实统计驱动）：

- 量型 p(t)：训练期平均分钟成交分布（`from_samples` 接口同样支持外部数据）；
- 冲击量级 Λ：平方根律 `成本/份 ≈ c_imp·σ_w·√(参与率)`，σ_w 为日盘窗口波动；
- 滑移/冲击占比：真实半价差 vs 总冲击 → 求解器 η；下限取论文规范占比
  （低于此进入 VWAP 基准博弈区，无约束解不可执行，由 long-only 钳制兜底）；
- **量纲归一化**：形变取 f = Λ·(v/X)^(1−ε)（参考速率 = 订单自身量级 X），
  整个形状问题精确映射为 share=1、速率 O(1) 的规范求解，扰动级数保持良态——
  这是从"论文量纲"到"实盘量纲"的关键一步。

**回放设计**：策略按分钟切片在真实分钟 VWAP 成交，叠加自身瞬态冲击
（propagator 因果传播）与线性滑移；VWAP 基准为含自身冲击的模型一致口径。
训练期 2024H2（125 天）建量型/校准，测试期 2025H1（117 天）逐日回放。

**结果**（117 个测试日的均值±标准差，单位 bp）：

| 品种 | 策略 | IS 口径 | VWAP 口径 |
|---|---|---|---|
| CU 沪铜 | **VWAP-opt** | 10.82±22.2 | **0.21**±10.3 |
| (X=3589 手, 5%) | **IS-opt** | **10.81**±23.0 | 0.40±9.0 |
| | TWAP | 11.50±28.5 | 1.39±5.1 |
| | naive-VWAP | 11.33±27.8 | 1.24±5.2 |
| | oracle-VWAP | 12.15±29.7 | 2.31±0.4 |
| BU 沥青 | **VWAP-opt** | 13.63±37.2 | **0.34**±12.1 |
| (X=7109 手, 5%) | **IS-opt** | **13.66**±38.3 | 0.62±9.9 |
| | TWAP | 15.19±44.8 | 2.68±5.4 |
| | naive-VWAP | 14.54±41.9 | 1.95±5.6 |
| | oracle-VWAP | 16.44±44.4 | 4.16±3.0 |

配对差（正 = 优化算法更省）：BU 上 naive−VWAP-opt = +1.61bp（t=2.0）、
TWAP−VWAP-opt = +2.34bp（t=2.3）、TWAP−IS-opt = +1.53bp（t=1.8）；
CU 上方向一致（+0.7～+1.2bp，t=1.1～1.6）。

**真实数据下的四点结论**：

1. **基准对角最优**：两品种均复现"各基准的最优算法在自身口径下期望滑点最低"；
2. **前载定理成立**：真实量型下 VWAP-opt 前半段 0.70、IS-opt 0.65、TWAP 0.50，
   与论文扰动级数的前载预测一致；VWAP 基准返还项使 VWAP-opt 比 IS-opt 更前载；
3. **完美量型 ≠ 最优**：oracle-VWAP（逐日按当日真实量型成交）期望滑点反而最高
   ——跟随市场尖峰放量执行会放大自身冲击与滑移，但它把滑点波动率压到 ±0.4bp
   （近乎完全对冲基准风险）；优化算法以少量基准风险（±10bp）换取期望成本下降
   约 1–4bp，期望-风险权衡清晰可见；
4. IS 口径的日间漂移噪声（±23–45bp）远大于策略间差异（~1bp），配对差检验
   是正确的统计口径；VWAP 口径则天然对冲了大部分漂移。

复现：`python3 scripts/validate_real.py [--products CU BU]`（约 1 分钟，
依赖 /data/ticker_data）；结果 JSON 与图输出至 `scripts/out/`。

## 7. v0.2：风险厌恶目标（库存风险项）

`ExecParams(risk=λ)` 在目标中加入 Almgren–Chriss 式库存风险项
`λ·∫(X−∫₀ᵗv)²dt`。其 Hessian 核 `2λ(T−max(s,t))` 对称半正定且与 ε 无关，
因此只并入线性算子与基线右端——**扰动递推结构零改动**，VWAP 交叉项与风险项
可同时启用。设计细节见 `docs/design/v0.2-risk-aversion.md`，迭代路线见
`docs/ITERATION_PLAN.md`，测试计划见 `docs/test-plan.md`。

**真实数据风险前沿**（`python3 scripts/validate_real.py --risk-grid 0.02 0.1 0.5`）：

| 品种 | 族 | λ=0 → 0.5 库存暴露 | IS 口径 std | VWAP 口径 std |
|---|---|---|---|---|
| CU | IS-opt | 0.2155 → 0.1857 | 23.0 → 21.5bp ↓ | 9.0 → 10.8bp ↑ |
| CU | VWAP-opt | 0.1986 → 0.1786 | 22.2 → 21.1bp ↓ | 10.3 → 11.6bp ↑ |
| BU | IS-opt | 0.2155 → 0.1857 | 38.2 → 36.5bp ↓ | 9.9 → 12.5bp ↑ |
| BU | VWAP-opt | 0.1968 → 0.1767 | 37.2 → 36.0bp ↓ | 12.1 → 13.9bp ↑ |

解读：λ 上升 → 前载增强 → **对到达价的持货漂移风险下降**（IS 口径 std，
理论保证单调），但**对市场 VWAP 的基准风险略升**（前载偏离市场量型）——
两种风险方向相反，风险预算须按口径分别设定。合成参数下（
`scripts/run_risk_frontier.py`）优化族在成本与库存暴露两个维度同时占优
TWAP/naive。有效域边界（VWAP 族强风险下线性基线转负）在设计中明确记录，
结构性解法（活跃集约束求解）排在 v0.3。

## 8. v0.3：日内滚动重规划

`execalgo/replanner.py` 的 `RollingPlanner` 每 K 分钟重解剩余窗口问题，状态
含剩余比例 φ、自体冲击残余 I_past（时间一致性必需）、日内量级信念 s_vol；
LU 按 (窗口长度, γ桶) 缓存，重规划为毫秒级。设计细节（含三个实现期确认的
关键推导）见 `docs/design/v0.3-rolling-replan.md`。

**实测发现**（v1.1 修正衔接项时刻单位 bug 后）：

1. **修订阻尼是必要的**：无阻尼的逐次重解在目标数值平坦方向游走，分钟级
   凸性把游走变成真实成本；α=0.5 的修订阻尼（向上一版计划混合）消除游走；
2. **量级通道的天花板**：滚动达成统计平价（CU −0.58bp t=−1.0）且 IS 口径
   风险下降（22.9→19.2bp），但无显著期望增益——归一化基准交叉项天然与
   市场量级无关，期望增益的主通道是尾部形状预测（v0.6 已证明不存在）；
3. **衔接项修复的连锁**：v0.3 发布时衔接项因单位 bug 形状失效（近似常数被
   乘子吸收），修复后 VWAP 子问题在常规日也触发正基线守卫（811/1505 次重规划），
   回退路径已升级为约束延拓（结构性 long-only）。

复现：`python3 scripts/validate_real.py --rolling`（日自适应冲击口径，
与默认口径的绝对数值不可直接对比）。

## 9. v0.4：敏感性与稳健性扫描（T13）

`execalgo/sensitivity.py` + `scripts/sensitivity_scan.py`：参与率 {1%, 5%, 10%} ×
冲击强度 c_imp {0.5, 1, 2} × 核衰减 {×1.5, ×1.0, ×0.75} × 品种 {CU, BU, MA, HC}
共 108 格，测试期 **2025 全年（243 天，H2 为纯留出）**。

| 结论 | 存活率 | 中位 t | 中位 bp |
|---|---|---|---|
| naive − VWAP-opt（VWAP 口径） | **108/108** | 2.2 | 1.34 |
| TWAP − IS-opt（IS 口径） | **108/108** | 1.7 | 0.97 |
| TWAP − VWAP-opt | **108/108** | 2.3 | 1.58 |
| 前载排序 0.5 < IS-opt < VWAP-opt | **108/108** | — | — |

四项核心结论在全部 108 格 100% 存活（阈值 80%），零正基线越界；优化边际随
参与率与冲击强度单调增强，最弱格为 MA@1%（+0.10bp，与 v0.5 拟合核的边际
缩减发现同向：小单、弱冲击、快衰减下塑形收益最小）。

## 10. v0.5：真实数据核校准

`execalgo/kernel_calibration.py` 把瞬态冲击核从论文先验（三指数拟合 `t^{-1/2}`）
换成数据驱动估计，识别链条为：符号响应（预交易基准 `m_{t−1}`，符号加权稳健于
持续订单流的重尾）→ **完整传播子去卷积**（前向同向流混合 + 预交易冲击态松弛的
双向结构；G(0) 由平滑外推闭合、超出窗长的尾部由参数族闭合迭代）→ 保完全单调的
三指数拟合。合成已知核+已知自相关的受控模拟验证了整条链（T14：C(1)=0.8 时
前 80 滞后恢复误差 ≤0.03）。

**真实数据结果**（CU/BU 训练 2024H2，`python3 scripts/calibrate_kernel.py`；
v1.1 修正 C(l) 分钟时基与 η 口径一致化后）：

| 量 | CU | BU | 论文核参照 |
|---|---|---|---|
| C(1) 符号自相关 | 0.069 | 0.064 | — |
| 拟合核半衰期(分钟) | 瞬时 / ~5 / 近永久 | 瞬时 / ~0.15 / 近永久 | 3.5 / 19.5 / 239.5 |
| 持续性 ∫G(120min) | 114.8 | 136.6 | 304.2 |
| 最优前载 IS（拟合/论文核） | 0.586 / 0.653 | 0.643 / 0.653 | — |

四点结论：

1. **中国期货分钟级订单流自相关很低**（C(1)≈0.07，与股票 0.2–0.4 显著不同），
   去卷积修正温和但仍是必要的识别步骤；
2. **经验冲击远不如论文核持续**——衰减更快、含近平永久分量；
3. **T15 方向确认**：核衰减更快 → 最优前载减弱（理论预期在真实数据成立）；
4. **对收益预期的重要校准**：在拟合核自洽定价下，v0.1 的优化边际缩小约一半
   （CU naive−VWAPopt +1.06→+0.37bp、BU +1.49→+1.15bp）但方向保持——
   冲击持续性越低，最优塑形的收益越小。论文核口径下的收益估计偏乐观，
   生产参数应采用数据驱动核。

## 11. v0.6：量型预测（负结果结案）

`execalgo/profile_forecast.py`：前缀条件化核估计器（前缀累积形状 + 竞价占比 +
夜盘量 + 星期效应），接入 `RollingPlanner(tail_forecaster=...)`；`realdata` 增加
夜盘量加载（文件 16–20 时 = 北京 21:00–次日 01:00，归因下一交易日）。合成双
regime 数据上机制验证至噪声地板。

**真实数据结论**（CU/BU 各 117 测试日，`python3 scripts/validate_real.py --forecast`；
v1.1 修正后 CU 复测 fcst −0.07bp t=−1.6 / oracle −0.04bp t=−0.4，结论不变）：

1. **日内量型形状不可预测**：所有条件化通道相对训练平均 0～−0.4%；训练平均
   已捕获跨日 TV 的 ~28%（两日间 TV 0.31/0.34 为无信息参照），其余为日间
   特异噪声——分钟级、块级、竞价、夜盘、星期、昨日形状全部无效；
2. **完美形状知识亦无均值价值**（v1.1 复测 CU −0.04bp t=−0.4，统计平价）——
   形状跟踪使执行向市场尖峰集中，分钟级凸性代价抵消基准返还收益；
3. 量型信息的残余价值在**基准风险**维度（与市场同步成交可把滑点波动率压低
   ~25 倍），不在期望成本。

执行算法的期望成本杠杆就此定位清楚：冲击核保真（v0.5）与价差捕获/约束
（v1.0），而非量型预测。

## 12. v1.0：盘口感知执行 + 约束套件（生产形态）

**约束求解**（[execalgo/constrained.py](execalgo/constrained.py)）：加边 KKT 系统上的
活跃集箱约束 `0 ≤ v ≤ cap`——long-only 钳制的结构化替代（v0.2 的博弈越界区现可
正确求解）与参与率上限（`ExecParams(upper_rate=...)` + `participation_upper()`）。
线性解 KKT 残差 ~1e-15；延拓路径为自由集投影牛顿（含三处实现期修复：merit 须取
自由集残差、牛顿步符号约定、√ 导数在边界 0 处的奇异性正则）。

**盘口执行**（[execalgo/execution_model.py](execalgo/execution_model.py)）：被动单按
对手方毛流量 × 队列捕获率成交、剩余吃单付半价差；竞价分钟 taker-only、停牌分钟
量顺延。加载器新增分钟买/卖毛流量，并过滤退化盘口（2025-04-07 跌停日 45% 快照
bid=0，未过滤会毒化 mid/价差使单日滑点虚增 ~1000bp）。

**真实数据结果**（CU 117 测试日，8%/min 参与率上限，`python3 scripts/run_production.py`）：

| 策略 | 旧模型 IS/VWAP (bp) | 盘口模型 IS/VWAP (bp) | 被动占比 |
|---|---|---|---|
| VWAP-opt（上限后） | 10.86 / 0.25 | 8.97 / **−1.64** | 49.0% |
| IS-opt（上限后） | 10.82 / 0.42 | 9.00 / −1.40 | 49.0% |
| TWAP | 11.50 / 1.39 | 10.00 / −0.12 | 49.6% |
| naive-VWAP | 11.33 / 1.24 | 9.67 / −0.41 | 49.2% |

三点结论：上限在 11–13% 分钟绑定、前载 0.696→0.682（约束路径有效且温和）；
盘口执行把旧口径 ~1.9bp 线性滑移摩擦变为 ≈0（被动占比 49% 时价差对冲），
优化器排序在两种执行模型下均保持；被动价差捕获使优化器在 VWAP 口径转为
**净负滑点**——执行质量（maker/taker 策略）成为比轨迹塑形更大的成本杠杆，
与 v0.5/v0.6 的杠杆定位结论闭环。

## 13. v1.2：队列/成交率标定

[execalgo/fill_calibration.py](execalgo/fill_calibration.py) 从 L5 深度数据
（bidVolume1-5 之和，Arrow 层聚合）标定被动成交概率曲线
`fill_fraction(flow/depth)`——非参数分箱均值 + 日块 bootstrap 90% CI。
`MakerPolicy(fill_curve=curve)` 接入 `replay_day_book`；不传曲线时行为与
v1.0 完全一致（向后兼容）。

**真实数据结果**（CU，`python3 scripts/run_v12.py`）：

| flow/depth 比 | <0.84 | 1.26 | 1.89 | 2.84 | 4.25 |
|---|---|---|---|---|---|
| fill_fraction | 0 | 0.20 | 0.46 | 0.64 | 0.76 |

| 执行策略 | 被动占比 | 价差捕获 (bp) |
|---|---|---|
| 常数 β=0.7 | 49.3% | −0.01 |
| **标定曲线** | **17.9%** | — |

**核心发现**：常数 β=0.7 **高估被动成交约 2 倍**——多数分钟卖流远小于
bid 深度，被动成交比 v1.0 假设的困难得多。标定后所有策略期望成本约升
0.3–0.4bp（更诚实的被动价差捕获估计），四策略排序保持不变。

## 14. v1.3：alpha 感知执行（框架验证）

`ExecParams(alpha=...)` 将期望回报信号注入 IS 目标泛函
（`J(v) -= ∫v·α dt`），`execalgo/signal.py` 提供 15 分钟动量信号与 IC 检验。

**真实数据结果**（CU 117 天，`python3 scripts/run_v13.py`）：

| 验收 | 结果 |
|---|---|
| T24 信号 IC | **−0.089（反转）**，正 IC 日 31.6% |
| T25 调度时机 | 高动量日**后载**（−0.0014）、低动量日**前载**（+0.0020）✓ |
| T26 成本差异 | −0.008bp (t=−0.4 不显著)，弱信号无净收益 |

三点结论：

1. **框架验证成功**：求解器正确响应信号方向（反转信号下高动量日后载）；
2. **CU 15 分钟动量是反转信号**（IC=−0.09），与市场微观结构均值回归一致；
3. **|IC|=0.09 的简单信号不足以产生正 alpha**（执行成本 > 时机收益）——
   alpha 感知执行的价值取决于信号质量而非框架；接入 LGBM 方向模型
   （更强信号 + 更长预测期）是提升路径。

**实现发现**：Newton 延拓对时间变化 RHS 极为敏感——逐分钟噪声 alpha
（std>0.02）导致发散；需日级方向聚合 + 光滑斜坡确保全日可解。

## 15. v1.4：跨品种冲击溢出 + 组合执行

[execalgo/cross_impact.py](execalgo/cross_impact.py) 从真实 L5 数据估计跨品种
冲击溢出（A 的符号 × B 的 log-mid 变化，bps），[scripts/run_v14.py](scripts/run_v14.py)
端到端验收。

**真实数据结果**（训练 2024H2，`python3 scripts/run_v14.py`）：

| 品种对 | 溢出比率 | R_AA(1) | R_AB(1) |
|---|---|---|---|
| CU→BC（同标的） | **1.041** | 1.66 | +1.71 |
| CU→ZNA（铜锌） | **0.978** | 1.66 | +1.58 |
| CU→BU（控制组） | **0.408** | 1.66 | +0.67 |

T29 独立 vs 顺序执行（CU+BC 80 天）：独立 12.86±26 bp < 顺序 15.15±31 bp
（独立−顺序 = −2.29bp, t=−3.1 显著）。

三点结论：

1. **溢出估计被共同因子污染**：控制组 CU→BU 溢出 0.408 远超独立品种预期
   （<0.1）——SHFE 品种共享宏观驱动，短窗口跨品种响应主要反映共同漂移
   而非真实的冲击传导；真正的溢出估计需控制共同因子（正交化或因子模型）；
2. **独立执行近优**：同标的 CU/BC 的独立执行显著优于分开半日执行——
   日内择时风险 >> 冲击重叠成本，在当前 5% 参与率量级下无需联合调度；
3. **后续方向**：更高参与率（>15%）或引入因子正交化后重新估计纯溢出。

## 16. v1.5：在线服务化（核心原语）

三个服务就绪原语（[execalgo/planner_state.py](execalgo/planner_state.py)、
[execalgo/decision.py](execalgo/decision.py)、`RollingPlanner.step()`）：

| 原语 | 接口 | 验收 |
|---|---|---|
| 增量步进 | `step(minute_vol, start, filled, fractions, pending) -> dict` | T30: 与批量执行逐位一致 |
| 状态快照 | `snapshot(planner, start, ...) -> PlannerSnapshot` | T31: 崩溃→恢复→继续 == 无中断 |
| 决策记录 | `make_decision(...) -> ExecutionDecision` | T32: 8 字段与参考架构对齐 |

部署时在服务侧组装（Redis stream 消费 + FastAPI + 生命周期管理），
参考 services/vwap-execution 的 coordinator/checkpoint 模式。

## 17.5 v1.6 整改冲刺（code review P0/P1/P2 修复）

修复影响已发布结论的 4 个 P0 + 7 个 P1 + 4 个高杠杆 P2（92 项全绿）：

| 修复 | 影响 |
|---|---|
| run_v12 profile 归一化 | v1.2 被动占比从 21.3%→**17.9%**（cross 项曾 225× 缩小） |
| run_v13 因果信号 | T26 配对差从 −0.03bp(t=−1.5)→**−0.008bp(t=−0.4)** |
| run_v13 IC 改训练期 | tilt_sign 现从训练期 IC 推导（无样本内配置） |
| run_v14 T29 诚实报告 | 注明回放无跨品种项，差异反映调度集中度而非溢出 |
| realdata 开盘前填 | 改用夜盘最后价（无前视） |
| branch eps 空梯 | eps_target < eps_step 时补一步而非返回线性基线 |
| step() forecaster | 加 today/night_volume 参数（在线模式等价性） |
| planner_state forecaster | 序列化 _live_night + today 用 ISO 日期 |
| fill_calibration 分箱 | 空箱 NaN + edge-based lookup |
| 证书路径 | alpha 时补 offset + cap-活跃节点排除 |

## 17.6 v2.0：LGBM 方向模型接入（信号弱瓶颈）

加载 Model-LightGBM 训练的 5 分钟方向模型（50 维 OHLCV 因子），
通过 `ExecParams(alpha=...)` 注入 alpha 感知执行框架。

| 验收 | 结果 |
|---|---|
| T33 信号 IC | **+0.070（正向）** vs 动量 −0.090（反转）✓ |
| T34 调度响应 | Newton 发散限制使响应不可测 ⚠️ |
| T35 成本差异 | +0.005bp (t=0.5)，不显著 ⚠️ |

**v2.3 修复后**：v2.2 存在三个 P0（lgbm_alpha 形状崩溃、夜盘价格键错误、
位移测试未按声明实现）。v2.3 修复后 run_v20 可正常运行：质量门正确拒绝
极端 alpha（29/30 天），唯一成功日方向正确（+0.02bp t=1.0）。

## 17.7 v2.4：盘口优化（tick 级队列物理）

执行成本的真实杠杆排序是 ①执行方式（maker/taker 价差 ~1.9bp）>
②冲击模型校准 > ③轨迹优化（vs TWAP 仅 0.7bp 不显著）。v2.4 聚焦 ①：
把分钟柱的 maker/taker 分割替换为 ~0.5s 快照上的显式微观结构模型。

**新增模块**：

- [execalgo/bookdata.py](execalgo/bookdata.py) — 单日 tick 级 L5 加载器
  （mid/microprice/逐档价量/tick-rule 符号量；开盘前锚点保住竞价量）
- [execalgo/tick_sim.py](execalgo/tick_sim.py) — 队列物理模拟器：
  L1 队尾入队（FIFO + 队首溢出部分成交 + 观测撤单钳制）、trade-through
  即刻成交、taker 走 L1-L5 逐档吃单（真实走册成本）、ladder 超时升级、
  半价差归一化的 microprice 门与撤单守卫
- `fill_calibration.py` v2.4 — `queue_basis="l1"`（L1 队列才是排队前方，
  L5 之和高估 ~8 倍）+ isotonic 单调平滑（消除 10 箱阶梯跳变）
- `execution_model.py` v2.4 — `passive_fraction="auto"`（去掉 π=0.5 策略
  硬顶，标定曲线是唯一限制）+ `spread_mode="minute"`（分钟级中位价差）

**真实数据结果**（CU 2025H1 117 天，TWAP，5% 参与率，
`python3 scripts/run_book_opt.py`）：

| 层 | 策略 | 被动占比 | vs taker 节省 (bp) | t 统计量 |
|---|---|---|---|---|
| 分钟 | v1.2 基线（π=0.5, L5, 日价差） | 15.7% | — | — |
| 分钟 | auto-π + L1 曲线 | 35.7% | +0.26 | — |
| tick | passive_fallback（整分钟挂单） | **79.1%** | **+0.881** | **52.7** |
| tick | ladder30（30s 超时升级） | 68.3% | +0.741 | 44.9 |
| tick | ladder30+gate | 57.6% | +0.654 | 31.2 |
| tick | ladder30+gate+guard | 46.7% | +0.508 | 26.5 |

**核心发现**：

1. 17.9% 被动占比的约束从来不是曲线偏差，而是 **π=0.5 策略上限与分钟柱
   粒度**——tick 级整分钟挂单直接到 79.1%，vs taker 节省 0.88bp/日且
   t=52.7（117 天配对）。
2. 被动成交的 5 分钟平均 drift 为 **+0.27bp（轻度有利，无系统性逆向选择
   惩罚）**；microprice 门与撤单守卫能改善 drift 的选择但被动占比损失大于
   drift 收益——**无条件挂单 + 日终强制补全**是当前性价比最高的盘口策略。
3. tick 级 +0.88bp ≈ 分钟级快赢（+0.26bp）的 3.4 倍，验证"粒度即价值"：
   队列位置、trade-through、走册成本都只在 tick 层可见。

**局限**（文档化假设）：模拟器不建模自身订单对盘口的反馈（5% 参与率下
单分钟切片 ~16 手 < L1 均值 ~25 手）；冲击项仍由 replay_day 定价，tick
层只度量微观执行。

## 17.8 v2.5：跨品种稳健性 + 双向对称

**跨品种复跑**（`run_book_opt.py --symbol {CU,BU,MA}`，各 117 测试日，
TWAP，5% 参与率）：

| 品种 | 半价差 | X (手) | fallback 被动占比 | 节省 (bp) | t | 门+守卫 被动/节省 |
|---|---|---|---|---|---|---|
| CU 铜 | 0.66bp | 3,589 | 79.1% | +0.881 | 52.7 | 46.7% / +0.51 |
| BU 沥青 | 1.46bp | 7,109 | 75.1% | +1.227 | 42.2 | 43.6% / +0.66 |
| MA 甲醇 | 2.00bp | 28,696 | 65.7% | +1.469 | 39.0 | 28.0% / +0.55 |

三个品种的结论完全一致：**被动占比随价差与订单规模递减（65-79%），
节省量与半价差成比例（0.9-1.5bp/日），策略排序不变**——无条件整分钟
挂单 > 守卫变体 > 门控变体 > 纯 taker。"无条件挂单 + 日终强制补全"是
跨 tick 制度稳健的结论（MA 参与量 127 手/分钟仍达 65.7%）。分钟层的
L1 曲线优势在 BU 上不成立（L5 略优，13.54 vs 13.65bp）——队列补充动态
品种相关，但该层只贡献 0.2-0.3bp，不影响 tick 层结论。2025-04-07 在
BU 上全天单边报价（跌停封死），被加载器的全天退化守卫正确跳过。

**卖出方向对称**（`LadderPolicy(side=-1)`）：挂 ask 被买方主动量消耗、
上方 trade-through、taker 走 bid L1-L5、门/守卫镜像（ask 侧钉死 = 上升
刀）。10 个 CU 日的双向对照：买入被动 81.4% / 节省 7.42 CNY，卖出被动
79.4% / 节省 7.47 CNY——经济镜像成立。`run_book_opt.py --side sell` 支
持卖出侧全量回测（成本与 drift 指标按方向归一）。

## 17. 目录结构

```
execalgo/            算法包
  model.py             多指数核 + 三指数拟合 t^{-1/2}（移植自参考实现）
  discretize.py        网格 / P,Q 算子 / 加边约束求解器（移植）
  impact.py            f(x;ε)=x^{1-ε} 及其一、二阶导
  series.py            扰动系数递推（支持 VWAP cross-term q）
  branch.py            牛顿延拓 / 平稳性 / Hessian（支持 q）
  vwap.py              市场分布（U 型 / 外部数据）+ 交叉项 q = X·Qp
  algorithms.py        ISAlgorithm / VWAPAlgorithm / ExecutionSchedule
  simulator.py         确定性期望成本 + 蒙特卡洛评估
  diagnostics.py       截断研究 / 收敛半径 / 前载分解
  realdata.py          真实期货数据：加载/时段映射/校准/回放回测
  replanner.py         v0.3 日内滚动重规划（RollingPlanner）
  profile_forecast.py  v0.6 量型预测（前缀核/平均/oracle）
  kernel_calibration.py v0.5 核校准（响应/去卷积/拟合/自举）
  sensitivity.py       v0.4 存活表聚合（survival_table/robust_verdict）
  constrained.py       v1.0 活跃集箱约束求解（线性精确 + 投影牛顿）
  execution_model.py   v1.0 盘口 maker/taker 执行模型（v1.2 fill_curve / v2.4 auto-π+分钟价差）
  fill_calibration.py  v1.2 被动成交概率标定（v2.4 L1 队列基线 + isotonic 平滑）
  signal.py            v1.3 动量信号 + IC 检验 + solver 单位缩放
  cross_impact.py      v1.4 跨品种冲击溢出估计（cross_response / spillover_ratio）
  planner_state.py     v1.5 快照/恢复（PlannerSnapshot + snapshot/restore）
  decision.py          v1.5 决策记录（ExecutionDecision + make_decision）
  lgbm_alpha.py        v2.0 LGBM 方向模型推理（25 因子×2 尺度→批量预测）
  bookdata.py          v2.4 单日 tick 级 L5 快照加载（~0.5s 分辨率）
  tick_sim.py          v2.4 tick 级队列物理模拟器（ladder/gate/guard 策略）
scripts/             演示与验证脚本（输出图与 JSON 到 scripts/out/）
tests/               pytest 套件（论文数值验收 + 回放不变量 + 加载器）
reference/           论文官方数值补充（只读对照，勿改）
```

## 18. 参考

- 微信文章《非线性市场冲击下的最优执行：扰动理论、前载现象与收敛边界》
- G. Ritter, *Perturbation theory for nonlinear transient impact* (2026)
- J. Gatheral, *No-dynamic-arbitrage and market impact* (2010)
- R. Almgren, N. Chriss, *Optimal execution of portfolio transactions* (2000)
- 参考实现: <https://github.com/gordonritter/perturbative-impact>（MIT）
- 真实数据: `/data/ticker_data`（L5 快照，加载约定参考 `/code/Model-LightGBM`）

# Post-training 微调技术对比总结

大语言模型的训练通常分为两大阶段：**预训练（Pre-training）**——在海量无标注文本上学习语言和世界知识；**后训练（Post-training）**——在预训练模型基础上，通过指令微调、参数高效微调、人类偏好对齐等手段，让模型更听话、更安全、更擅长特定任务。本文梳理几类主流 post-training 技术，并做横向对比。

---

## 1. 整体分类

Post-training 技术大致可分为两条主线：

| 类别 | 目标 | 代表方法 |
|---|---|---|
| **参数高效微调（PEFT）** | 用更少的显存/算力做全量或指令微调 | LoRA、QLoRA、DoRA、Adapter、Prefix-Tuning |
| **偏好对齐 / 强化学习类方法** | 让模型输出更符合人类（或规则）偏好 | PPO（RLHF）、DPO、GRPO、KTO、ORPO、SimPO、RLAIF/RLVR |

两条主线并不互斥——实践中常见的组合是：**SFT（全量或 LoRA）→ 偏好对齐（DPO/PPO/GRPO）**，逐层递进对齐模型行为。

---

## 2. 参数高效微调（PEFT）系列

### 2.1 LoRA（Low-Rank Adaptation, 2021）

- **核心思路**：冻结原始权重矩阵 `W₀ ∈ ℝ^{d×k}`，为其注入两个低秩矩阵 `B ∈ ℝ^{d×r}`、`A ∈ ℝ^{r×k}`（`r ≪ min(d,k)`），训练时只更新 `A`、`B`：

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA
$$

  对应的前向传播变为：

$$
h = W_0 x + \frac{\alpha}{r} BA\, x
$$

  其中 `α` 是缩放系数，`r` 是秩。初始化时通常 `A` 随机初始化、`B` 置零，保证训练起点 `ΔW = 0`，不改变原模型行为。

- **可训练参数量**：原本全量微调需要更新 `d×k` 个参数，LoRA 只需 `r×(d+k)` 个，当 `r` 远小于 `d, k` 时压缩比非常可观，例如 `d=k=4096, r=8` 时参数量降至约 0.4%。
- **优点**：多个任务的 LoRA 权重可以即插即用、方便切换和合并（推理时直接把 `ΔW` 加回 `W₀` 即可，不引入额外推理延迟）。
- **局限**：秩 `r` 的选择需要权衡表达能力与开销；对于差异很大的任务，效果通常不如全量微调。

### 2.2 QLoRA（2023）

- **核心思路**：在 LoRA 基础上，把基座模型量化为 **4-bit NF4** 格式冻结存储，仅在计算时反量化，再叠加 LoRA 适配器训练。

- **量化的基本形式**：对权重张量 `W`，用一个缩放常数（absmax）把浮点值映射到有限个离散层级：

$$
c = \frac{\max(|W|)}{2^{k-1}-1}, \qquad W_{k\text{-bit}} = \mathrm{round}\!\left(\frac{W}{c}\right)
$$

  前向计算前需要先反量化回浮点：

$$
\hat{W} = c \cdot W_{k\text{-bit}}
$$

  QLoRA 使用的 NF4（4-bit NormalFloat）并非均匀量化，而是按标准正态分布的分位数设计量化层级，使得量化误差在权重密集区域（接近 0）更小。

- **双重量化（Double Quantization）**：把上面的缩放常数 `c` 本身也看作一组数值，再做一次 8-bit 量化：

$$
c_{\text{8bit}} = \mathrm{round}\!\left(\frac{c}{c_2}\right), \qquad c_2 = \frac{\max(|c|)}{2^{7}-1}
$$

  每个参数额外的存储开销从约 32/64 位量化常数降到不到 1 bit（分块共享）。

- **训练时的整体前向**（结合 LoRA）：

$$
h = \hat{W}_0\, x + \frac{\alpha}{r} BA\, x
$$

  其中 `Ŵ₀` 是 4-bit 反量化得到的基座权重（冻结、不参与梯度更新），只有 `A, B` 参与反向传播，梯度和优化器状态都只需保存在 LoRA 部分，显存占用大幅降低。

- **分页优化器（Paged Optimizers）**：利用 NVIDIA 统一内存，在显存峰值时把优化器状态换出到 CPU 内存，避免因梯度尖峰导致 OOM。
- **意义**：让消费级单卡（如 24GB 显存）也能微调 30B~65B 级别的模型，是资源受限场景下的首选方案。

### 2.3 其他值得一提的 PEFT 方法

| 方法 | 特点 |
|---|---|
| **DoRA**（Weight-Decomposed LoRA） | 将权重分解为“方向+幅度”，只对方向做低秩更新，效果常优于原始 LoRA |
| **Prefix-Tuning / P-Tuning v2** | 不改权重，而是在输入前插入可学习的虚拟 token（soft prompt） |
| **(IA)³** | 通过可学习的缩放向量对激活值做逐元素缩放，参数量比 LoRA 更小 |
| **Adapter Tuning** | 在每层插入小型瓶颈 MLP 模块，是 PEFT 概念的早期代表 |

---

## 3. 偏好对齐 / 强化学习类方法

### 3.1 背景：RLHF 三段式流程

传统 RLHF（Reinforcement Learning from Human Feedback）分三步：

1. **SFT**：用高质量指令数据做监督微调，得到基础对话模型。
2. **训练奖励模型（Reward Model, RM）**：用人类标注的偏好对（chosen vs. rejected）训练一个打分模型。
3. **RL 阶段（通常用 PPO）**：用 RM 的打分作为奖励信号，通过强化学习优化策略模型。

这个流程效果好但**链条长、成本高、训练不稳定**，后续的 DPO、GRPO 等方法都是针对其痛点的改进。

### 3.2 PPO（Proximal Policy Optimization）

- **角色配置**：需要同时维护 **策略模型（Actor）** `π_θ`、**价值模型（Critic）** `V_φ`、**奖励模型（RM）**、以及一个冻结的**参考模型** `π_ref`，四个模型同时占用显存。

- **奖励信号**：RLHF 中真实优化的奖励并非 RM 的原始打分，而是加了 KL 惩罚项、防止策略偏离参考模型太远：

$$
r(x,y) = r_{\text{RM}}(x,y) - \beta \log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

- **裁剪目标函数**：设概率比 `r_t(θ) = π_θ(a_t|s_t) / π_{θ_{\text{old}}}(a_t|s_t)`，PPO 的核心是限制每步更新幅度：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta) \hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t\big)\Big]
$$

  其中 `Â_t` 是优势函数（通常用 GAE 从 Critic 的价值估计中计算），`ε` 是裁剪范围（如 0.2）。当新旧策略差异过大时，裁剪会截断梯度，避免单步更新过猛导致训练崩溃。

- **优点**：理论成熟、上限高，OpenAI 的 InstructGPT/ChatGPT 均采用此路线。
- **缺点**：工程复杂度高、超参数敏感、显存和计算开销大（同时需要 Actor+Critic+RM+参考模型），训练稳定性依赖大量调参经验。

### 3.3 DPO（Direct Preference Optimization, 2023）

- **核心思路**：跳过显式奖励模型训练和 RL 采样过程，直接从偏好对 `(prompt, chosen, rejected)` 出发，推导出一个可以直接用监督学习方式优化的损失函数。

- **推导起点**：假设奖励与策略之间存在如下解析关系（Bradley-Terry 偏好模型 + KL 约束下的最优策略形式）：

$$
r(x,y) = \beta \log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$

  代入 Bradley-Terry 偏好概率 `P(y_w \succ y_l \mid x) = \sigma(r(x,y_w) - r(x,y_l))`，配分函数 `Z(x)` 相减后消去，得到可直接优化的损失：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

  其中 `y_w`（winner/chosen）、`y_l`（loser/rejected）是同一个 prompt 下的偏好对，`β` 控制对参考模型的偏离惩罚强度，`σ` 是 sigmoid 函数。整个损失只涉及策略模型和冻结的参考模型两者的对数概率，形式上等价于一个二分类的交叉熵损失。

- **优点**：
  - 无需训练/维护奖励模型和价值模型，只需要策略模型 + 一个冻结的参考模型；
  - 训练过程类似普通监督学习，稳定、易实现、显存开销远低于 PPO。
- **局限**：效果上限依赖离线偏好数据的质量和覆盖面，无法像在线 RL 一样持续探索新的输出分布。
- **地位**：目前工业界做偏好对齐的**性价比首选**，Llama、Zephyr、Qwen 等多个开源模型系列都采用或提供 DPO 版本。

### 3.4 GRPO（Group Relative Policy Optimization，DeepSeekMath / DeepSeek-R1, 2024）

- **核心思路**：保留 PPO 的在线 RL 框架，但**去掉价值模型（Critic）**：对同一个 prompt `x` 用旧策略 `π_{θ_{\text{old}}}` 采样一组（group）输出 `{o_1, ..., o_G}`，各自获得奖励 `{r_1, ..., r_G}`，用**组内**均值和标准差做归一化，直接得到相对优势，替代 Critic 估计的 Advantage：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

- **目标函数**：在组内对每个输出的每个 token 做类似 PPO 的裁剪更新，并显式加入相对参考模型的 KL 惩罚（不再隐藏进奖励里，而是单独作为一项）：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\Big(r_i(\theta)\,\hat{A}_i,\ \mathrm{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_i\Big)\right] - \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

  其中 `r_i(θ) = π_θ(o_i|x) / π_{θ_old}(o_i|x)`。由于 Advantage 直接来自组内统计量，不再需要单独训练一个和策略模型同规模的 Critic 网络。

- **优点**：
  - 显存和计算开销比 PPO 显著降低（少一个和策略模型同规模的价值网络）；
  - 常与**可验证奖励（Rule-based / Verifiable Reward，RLVR）**配合——例如数学题答案对错、代码是否通过测试给出 `r_i ∈ {0,1}` 这样的规则奖励——不依赖学习出来的奖励模型，奖励信号更客观、抗 reward hacking。
- **代表应用**：DeepSeek-R1 等推理模型的强化学习训练阶段，在数学、代码等有明确对错标准的任务上效果突出，是目前"强推理模型"训练的主流方法之一。

### 3.5 其他重要的对齐方法

| 方法 | 一句话概括 |
|---|---|
| **KTO**（Kahneman-Tversky Optimization） | 不需要成对偏好数据，只需要单条样本的"好/坏"二元标签即可训练，基于行为经济学的前景理论建模人类对损失/收益的非对称敏感度 |
| **ORPO**（Odds Ratio Preference Optimization） | 把 SFT 和偏好优化**合并成一个阶段**，用胜率比（odds ratio）惩罚项直接在 SFT loss 上附加偏好约束，省去参考模型 |
| **SimPO** | 去掉参考模型，用**长度归一化后的隐式奖励**代替 DPO 中的对数概率比，简化实现同时缓解模型偏爱啰嗦输出的问题 |
| **RLAIF / Constitutional AI** | 用强模型（或规则）自动生成偏好标注代替人工标注，降低对齐数据的人力成本，常与 DPO/PPO 配合使用 |
| **RLVR**（Reinforcement Learning with Verifiable Rewards） | 不是一种独立算法，而是一类"奖励来源"——用可自动验证的规则（答案对错、单测通过率）代替人工/模型打分，常与 GRPO/PPO 结合，是推理模型训练的关键要素 |

**关键公式补充**：

- **KTO** 借鉴前景理论，把参考点设为当前批次的期望 KL 散度 `z_0 = D_{\mathrm{KL}}(\pi_\theta(y'|x) \,\|\, \pi_{\text{ref}}(y'|x))`，对期望/非期望样本分别用不同的效用函数：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)}\Big[\lambda_{y} - v(x,y)\Big], \quad
v(x,y) =
\begin{cases}
\sigma\big(\beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0\big), & y \text{ 为期望输出} \\[4pt]
\sigma\big(z_0 - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\big), & y \text{ 为非期望输出}
\end{cases}
$$

  这样即可用非成对的单条“好/坏”标签训练，无需构造 `(y_w, y_l)` 偏好对。

- **ORPO** 在标准 SFT 交叉熵损失上，叠加一个基于胜率比（odds）的惩罚项，让模型在学习生成 `y_w` 的同时压低生成 `y_l` 的相对概率。定义 `y` 的胜率为 `\mathrm{odds}_\theta(y|x) = \dfrac{p_\theta(y|x)}{1-p_\theta(y|x)}`：

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \Big(-\log \sigma\big(\log \mathrm{odds}_\theta(y_w|x) - \log \mathrm{odds}_\theta(y_l|x)\big)\Big)
$$

  整个过程不引入参考模型 `π_ref`，SFT 和偏好对齐在同一阶段完成。

- **SimPO** 去掉参考模型，直接用**长度归一化的平均对数概率**作为隐式奖励 `r(x,y) = \dfrac{\beta}{|y|}\log \pi_\theta(y|x)`，并引入目标间隔 `γ`：

$$
\mathcal{L}_{\text{SimPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]
$$

  长度归一化避免了 DPO 中模型倾向于生成更长回答以“稀释”负对数概率的偏差。

---

## 4. 横向对比表

| 维度 | LoRA/QLoRA | PPO | DPO | GRPO |
|---|---|---|---|---|
| 所属类别 | 参数高效微调 | RL 对齐 | 离线对齐 | RL 对齐 |
| 是否需要奖励模型 | 不涉及 | 需要 | 不需要（数据自带偏好标签） | 可选（常配合规则奖励） |
| 是否需要价值/Critic模型 | 不涉及 | 需要 | 不需要 | 不需要 |
| 同时驻留的模型数 | 1（基座+适配器） | 4（策略/价值/奖励/参考） | 2（策略/参考） | 2~3（策略/参考/可选奖励） |
| 显存开销 | 很低 | 高 | 中 | 中低 |
| 训练稳定性 | 高 | 较低，需精细调参 | 高 | 中高 |
| 数据形式 | 指令-回答对 | prompt + 奖励信号（在线采样） | (prompt, chosen, rejected) 离线对 | prompt + 组内多次采样的奖励 |
| 典型场景 | 资源受限下的 SFT/领域适配 | 追求上限、有工程能力的对齐 | 性价比最高的偏好对齐 | 数学/代码等可验证任务的推理能力强化 |

---

## 5. 如何选择：一个简单的决策思路

1. **只是想让模型学会某个任务/风格，显存有限** → 用 **QLoRA**（甚至 LoRA）做 SFT。
2. **有人工标注的偏好对数据，想让模型更"对齐人类喜好"，且希望流程简单稳定** → 用 **DPO**（或其变体 SimPO/ORPO）。
3. **只有单条"好/坏"标签，没有成对比较数据** → 用 **KTO**。
4. **任务有明确对错标准（数学、代码、Agent 工具调用等），想进一步压榨推理能力，且能接受在线 RL 的工程复杂度** → 用 **GRPO + RLVR**。
5. **有充足工程资源、追求效果上限，且能承担奖励模型训练与 RL 调参成本** → 传统 **PPO/RLHF** 仍然是效果天花板较高的选择。

实践中，头部开源模型（如 DeepSeek、Qwen、Llama 系列）常见的组合路径是：

**预训练 → （LoRA/全量）SFT → DPO 做通用偏好对齐 → GRPO+RLVR 强化特定推理能力**

这条路径兼顾了训练成本、稳定性和最终效果，也是目前 post-training 领域的主流范式。

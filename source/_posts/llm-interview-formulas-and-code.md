---
title: "大模型面试常用公式与手撕代码"
date: 2026-08-27 21:10:00
updated: 2026-08-27 21:10:00
description: "覆盖 Transformer、激活函数、归一化、优化器、位置编码、PPO、GRPO、DPO、DAPO，以及 MHA、GQA、RoPE、RMSNorm、SwiGLU 的 PyTorch 手撕实现。"
categories:
  - 面试
  - 面试准备
tags:
  - 大模型
  - 公式
  - PyTorch
  - 强化学习
math: true
---

> 目标：看到公式能说清**每一项是什么、解决什么问题、代价是什么**；看到代码能从张量形状开始写，并主动处理 mask、因果性和数值稳定性。
>
> 约定：batch size 为 $B$，序列长度为 $T$，隐藏维度为 $d$，Query 头数为 $H_q$，KV 头数为 $H_{kv}$，单头维度为 $d_h=d/H_q$，词表大小为 $V$。

## 0. 面试前 10 分钟速记

| 主题 | 必须能脱口而出的式子或结论 |
|---|---|
| Softmax | $p_i=\exp(z_i-m)/\sum_j\exp(z_j-m)$，减最大值 $m$ 防溢出 |
| 交叉熵 | $\mathcal L=-\sum_i y_i\log p_i$；one-hot 时就是正确类别的 NLL |
| Attention | $\operatorname{softmax}(QK^\top/\sqrt{d_h}+M)V$ |
| GQA | $H_q$ 个 Q 头共享 $H_{kv}$ 组 K/V；每组服务 $H_q/H_{kv}$ 个 Q 头 |
| RoPE | 对 Q/K 的二维通道对做位置相关旋转，使内积只显式依赖相对位置 |
| RMSNorm | $x/\sqrt{\operatorname{mean}(x^2)+\epsilon}\odot\gamma$，不减均值 |
| SwiGLU | $\operatorname{SiLU}(xW_g)\odot(xW_u)$ 再做下投影 |
| AdamW | Adam 自适应更新与 weight decay 解耦 |
| PPO | 用 clip 限制新旧策略概率比，避免单次更新过大 |
| GRPO | 同一 prompt 采样一组回答，用组内相对奖励代替 critic |
| DPO | 直接提高 chosen 相对 rejected 的隐式奖励差，不需要 rollout/critic |
| DAPO | GRPO 路线上的 Clip-Higher、动态采样、token 级 loss、过长奖励塑形 |

---

## 1. 概率、Softmax 与常用损失

### 1.1 Softmax、温度与数值稳定

给定 logits $z\in\mathbb R^V$：

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_{j=1}^{V}\exp(z_j/\tau)}.
$$

$\tau$ 是温度。$\tau<1$ 让分布更尖锐，$\tau>1$ 让分布更平坦；$\tau\to0$ 接近 argmax。实际计算必须使用：

$$
p_i=\frac{\exp((z_i-m)/\tau)}{\sum_j\exp((z_j-m)/\tau)},\qquad m=\max_j z_j.
$$

对应的 log-sum-exp 技巧：

$$
\operatorname{LSE}(z)=m+\log\sum_j\exp(z_j-m).
$$

**常见追问：为什么除以 $\sqrt{d_h}$？** 若 Q、K 各维独立、零均值、单位方差，点积方差随 $d_h$ 增长。缩放后方差约保持常数量级，避免 Softmax 过早饱和、梯度过小。

### 1.2 信息熵、交叉熵、KL 散度

$$
H(p)=-\sum_i p_i\log p_i,
$$

$$
H(p,q)=-\sum_i p_i\log q_i,
$$

$$
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}=H(p,q)-H(p).
$$

KL 不对称，也不满足三角不等式，因此不是距离。训练时真实分布 $p$ 固定，最小化交叉熵等价于最小化 $D_{\mathrm{KL}}(p\|q)$。

one-hot 标签 $y$ 的交叉熵：

$$
\mathcal L_{\mathrm{CE}}=-\sum_{i=1}^{V}y_i\log p_i=-\log p_{y}.
$$

标签平滑把目标改为：

$$
y_i^{\mathrm{LS}}=(1-\epsilon)y_i+\frac{\epsilon}{V}.
$$

它能缓解过度置信，但会改变概率校准；大模型 SFT 中是否使用要结合数据噪声和任务需求验证。

### 1.3 自回归语言建模与困惑度

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),
$$

$$
\mathcal L_{\mathrm{NTP}}=-\frac{1}{N}\sum_{t=1}^{T}m_t\log p_\theta(x_t\mid x_{<t}),
\qquad N=\sum_t m_t.
$$

$m_t\in\{0,1\}$ 是 loss mask。SFT 通常只让 assistant 输出位置参与损失；padding、system/user/tool 是否参与必须显式约定。

$$
\operatorname{PPL}=\exp(\mathcal L_{\mathrm{NTP}}).
$$

PPL 只能在**相同 tokenizer、相同数据和相同 token 聚合口径**下比较。不同词表会改变 token 数和每 token 难度。

### 1.4 二分类、Focal Loss 与对比学习

二元交叉熵：

$$
\mathcal L_{\mathrm{BCE}}=-\left[y\log p+(1-y)\log(1-p)\right].
$$

类别不均衡时常见 Focal Loss：

$$
\mathcal L_{\mathrm{focal}}=-\alpha_t(1-p_t)^\gamma\log p_t.
$$

InfoNCE（以样本 $i$ 的正对 $i^+$ 为例）：

$$
\mathcal L_i=-\log
\frac{\exp(\operatorname{sim}(h_i,h_{i^+})/\tau)}
{\sum_j\exp(\operatorname{sim}(h_i,h_j)/\tau)}.
$$

增大 batch 往往能提供更多负样本，但也会增加假负样本概率、通信和显存成本。

---

## 2. Transformer 核心公式

### 2.1 Scaled Dot-Product Attention

输入 $X\in\mathbb R^{B\times T\times d}$：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

因果 mask 中：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

训练时注意力分数的时间、空间复杂度都是 $O(T^2)$ 量级；自回归解码配合 KV Cache 后，每一步只计算新 token 的 Q/K/V，但新 Q 仍需读取历史 K/V。

### 2.2 MHA、MQA 与 GQA

MHA 将每个头独立计算后拼接：

$$
\operatorname{MHA}(X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{H_q})W_O,
$$

$$
\operatorname{head}_h=\operatorname{Attn}(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)}).
$$

三种结构的 KV 头数：

| 结构 | KV 头数 | 关系 | 主要取舍 |
|---|---:|---|---|
| MHA | $H_{kv}=H_q$ | 每个 Q 头有自己的 K/V | 表达强，KV Cache 最大 |
| GQA | $1<H_{kv}<H_q$ | 每 $g=H_q/H_{kv}$ 个 Q 头共享一组 K/V | 质量和带宽折中 |
| MQA | $H_{kv}=1$ | 所有 Q 头共享 K/V | KV 最小，容量更敏感 |

单层 KV Cache 的元素数量约为：

$$
N_{KV}=2BT H_{kv}d_h.
$$

若每个元素占 $s$ bytes、共有 $L$ 层：

$$
\operatorname{Memory}_{KV}\approx2BLTH_{kv}d_hs.
$$

因此相同 $d_h$ 下，GQA 相比 MHA 的 KV Cache 比例约为 $H_{kv}/H_q$。注意：GQA 降低的是 K/V 存储和 decode 带宽，Q 头数与输出维度并未同比减少。

### 2.3 残差与 Pre-Norm

现代 decoder-only 模型通常采用 Pre-Norm：

$$
X'=X+\operatorname{Attention}(\operatorname{Norm}(X)),
$$

$$
Y=X'+\operatorname{FFN}(\operatorname{Norm}(X')).
$$

Pre-Norm 为梯度提供更直接的残差通路，深层训练通常更稳定；Post-Norm 可写为 $\operatorname{Norm}(X+F(X))$，深层模型往往更依赖 warmup 和初始化技巧。

### 2.4 输出层与权重绑定

$$
z_t=h_tW_{\mathrm{lm}}+b,qquad p(x_{t+1}|x_{\le t})=\operatorname{softmax}(z_t).
$$

若输入 embedding 与 LM head 权重绑定，则 $W_{\mathrm{lm}}=E^\top$，可以减少约 $Vd$ 个参数，并让输入、输出词向量共享语义空间。

---

## 3. 位置编码

### 3.1 正弦位置编码

原始 Transformer 使用：

$$
\operatorname{PE}(p,2i)=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
$$

它不增加可学习参数，并可由三角恒等式表达相对位移；但现代 decoder LLM 更常见 RoPE。

### 3.2 RoPE

把相邻两维视作一个二维向量。位置 $m$、第 $i$ 个频率的旋转为：

$$
R_{m,i}=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix},
\qquad
\theta_i=\operatorname{base}^{-2i/d_h}.
$$

$$
q_m'=R_mq_m,\qquad k_n'=R_nk_n.
$$

核心性质：

$$
(q_m')^\top k_n'=q_m^\top R_{n-m}k_n.
$$

因此注意力内积显式依赖相对位置 $n-m$。RoPE 只旋转 Q/K，不旋转 V。长上下文外推常调整频率或位置尺度，但“窗口能接收更长输入”不等于“模型学会利用远距离信息”，仍需长上下文训练与评测。

### 3.3 ALiBi

ALiBi 不修改 Q/K，而是给不同注意力头加入与距离成正比的偏置：

$$
S_{ij}^{(h)}=\frac{q_i^{(h)}(k_j^{(h)})^\top}{\sqrt{d_h}}-m_h(i-j),\qquad j\le i.
$$

它简单、无需位置 embedding，具有一定长度外推性；线性距离惩罚也可能压低非常远的依赖。

---

## 4. 激活函数、FFN 与归一化

### 4.1 常见激活函数

| 名称 | 公式 | 特点 |
|---|---|---|
| Sigmoid | $\sigma(x)=1/(1+e^{-x})$ | 输出 $(0,1)$；大正负区间梯度饱和 |
| Tanh | $\tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})$ | 零中心；仍有饱和问题 |
| ReLU | $\max(0,x)$ | 简单稀疏；负区间可能“死亡” |
| LeakyReLU | $\max(\alpha x,x)$ | 给负区间保留小梯度 |
| GELU | $x\Phi(x)$ | 平滑门控；BERT/GPT-2 常见 |
| SiLU/Swish | $x\sigma(x)$ | 平滑、非单调；现代 LLM 常见 |

GELU 常用近似：

$$
\operatorname{GELU}(x)\approx\frac{x}{2}
\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right].
$$

### 4.2 GLU 与 SwiGLU

普通 FFN：

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2.
$$

SwiGLU：

$$
\operatorname{SwiGLU}(x)=
\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d.
$$

它用一条分支产生门值，另一条分支提供内容，逐元素相乘后下投影。为保持与普通 $4d$ FFN 接近的参数量，SwiGLU 的中间维度常取约 $\frac{8}{3}d$ 后再向硬件友好倍数取整。

### 4.3 LayerNorm 与 RMSNorm

对单个 token 的隐藏向量 $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_i x_i,qquad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LayerNorm}(x)=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot\gamma+\beta.
$$

RMSNorm 不做中心化：

$$
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma.
$$

RMSNorm 少计算均值和 bias，工程上更简单。两者都是沿隐藏维归一化每个 token，不依赖 batch 统计，因此适合变长序列和小 batch。

---

## 5. 优化器与训练稳定性

### 5.1 SGD、Momentum

$$
\theta_t=\theta_{t-1}-\eta g_t,
\qquad g_t=\nabla_\theta\mathcal L(\theta_{t-1}).
$$

Momentum：

$$
v_t=\mu v_{t-1}+g_t,
\qquad
\theta_t=\theta_{t-1}-\eta v_t.
$$

动量积累一致方向、抑制来回震荡。不同框架对 $v_t$ 的缩放约定可能不同，面试时说明思想即可。

### 5.2 Adam

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

一阶、二阶矩初始化为 0，前期会偏向 0，所以需要 bias correction。Adam 为每个参数维护一阶矩、二阶矩，若状态用 FP32，状态本身约占每参数 8 bytes，不含参数、梯度和 master weights。

### 5.3 AdamW

AdamW 将权重衰减与自适应梯度解耦：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

若把 $L_2$ 正则直接加到 loss，梯度项 $\lambda\theta$ 还会被 Adam 的二阶矩缩放，不再等价于统一比例的 weight decay。bias 和 normalization scale 通常不做 weight decay。

### 5.4 Muon

Muon 主要用于 Transformer 中的二维权重矩阵。若动量矩阵的奇异值分解为 $M=U\Sigma V^\top$，其理想的正交化更新方向是矩阵符号/极分解因子：

$$
\operatorname{msign}(M)=UV^\top.
$$

实际不会直接做昂贵的 SVD，而是用 Newton-Schulz 迭代近似矩阵正交化，再按矩阵形状缩放并更新权重。直觉是把更新矩阵过大或过小的奇异值拉到更接近的尺度，改善大矩阵更新的条件数。

Muon 不适合无脑用于所有参数：embedding、LM head、norm、bias、标量/向量参数通常保留 AdamW；融合 QKV 或 GLU 矩阵还应按语义块拆分。分布式实现需要同时考虑矩阵重组、通信量与各 rank 的正交化负载。

### 5.5 学习率与梯度裁剪

线性 warmup + cosine decay 的一种写法：

$$
\eta_t=\begin{cases}
\eta_{\max}\frac{t}{T_w},&t<T_w,\\
\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right],&t\ge T_w.
\end{cases}
$$

全局梯度范数裁剪：

$$
g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).
$$

它限制极端 step 的更新，但不能掩盖持续的 loss spike；仍要排查异常数据、学习率、精度溢出、MoE 路由和并行一致性。

---

## 6. 参数高效微调与蒸馏

### 6.1 LoRA

冻结原权重 $W_0\in\mathbb R^{d_{out}\times d_{in}}$，只训练低秩增量：

$$
W=W_0+\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA,
$$

其中 $A\in\mathbb R^{r\times d_{in}}$、$B\in\mathbb R^{d_{out}\times r}$、$r\ll\min(d_{in},d_{out})$。

$$
y=W_0x+\frac{\alpha}{r}B(Ax).
$$

新增参数量为 $r(d_{in}+d_{out})$，而全量矩阵参数为 $d_{in}d_{out}$。$r$ 控制增量矩阵最大秩，$\alpha/r$ 控制更新尺度；两者不能只凭经验孤立比较。

### 6.2 知识蒸馏

温度为 $\tau$ 的 logits 蒸馏：

$$
\mathcal L_{KD}=\tau^2D_{\mathrm{KL}}
\left(\operatorname{softmax}(z_T/\tau)\,\|\,
\operatorname{softmax}(z_S/\tau)\right).
$$

$\tau^2$ 用于补偿 Softmax 温度导致的梯度尺度变化。实践中常与 hard-label CE 混合：

$$
\mathcal L=(1-\lambda)\mathcal L_{CE}+\lambda\mathcal L_{KD}.
$$

---

## 7. 强化学习与偏好优化

### 7.1 Policy Gradient、Baseline 与 GAE

策略目标：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{T}\gamma^tr_t\right].
$$

REINFORCE：

$$
\nabla_\theta J(\theta)=
\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)G_t\right].
$$

减去与 action 无关的 baseline 不改变期望，但能降方差：

$$
A_t=Q(s_t,a_t)-V(s_t).
$$

GAE：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

$$
\hat A_t^{\mathrm{GAE}}=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$ 越大，偏差通常更小、方差更大。LLM 的结果奖励常只在序列末尾给出，token-level credit assignment 因而更难。

### 7.2 PPO

新旧策略概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
=\exp\left(\log\pi_\theta-\log\pi_{\theta_{old}}\right).
$$

Clipped surrogate objective：

$$
L_{\mathrm{clip}}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

训练通常最小化以下组合损失：

$$
\mathcal L_{\mathrm{PPO}}=
-L_{\mathrm{clip}}+c_v\mathcal L_V-c_eH(\pi_\theta)
+\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{ref}).
$$

| 项 | 作用 |
|---|---|
| clip | 限制新旧策略差异，抑制破坏性更新 |
| value loss | 训练 critic 估计回报，降低 advantage 方差 |
| entropy bonus | 防止策略过早坍缩，鼓励探索 |
| reference KL | 防止语言策略偏离 SFT/reference 模型过远 |

**高频坑：**clip 不是直接裁剪参数或梯度；它裁剪的是 action 概率比。PPO 仍是 on-policy 风格，旧 rollout 反复训练太久会导致策略分布失配。

### 7.3 GRPO

对同一个 prompt $q$，旧策略采样 $G$ 个回答 $o_1,\ldots,o_G$，得到奖励 $R_1,\ldots,R_G$。组内标准化 advantage：

$$
\hat A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}.
$$

常见 token 级 clipped objective：

$$
J_{\mathrm{GRPO}}(\theta)=
\mathbb E\left[
\frac1G\sum_{i=1}^{G}\frac1{|o_i|}
\sum_{t=1}^{|o_i|}
\left\{
\min\left(
r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right)
-\beta D_{\mathrm{KL}}\left(
\pi_\theta(\cdot|s_{i,t})\|\pi_{ref}(\cdot|s_{i,t})
\right)
\right\}
\right].
$$

GRPO 用同一 prompt 下的相对奖励作为 baseline，省掉 critic/value model，但**不省 rollout**，通常还会因每题多采样增加生成成本。

组内奖励全相等时 $\hat A_i\approx0$，该 prompt 几乎不产生学习信号；奖励非常离散时，小组标准差也会让梯度尺度敏感。实现中要对零方差加 $\epsilon$，并监控有效组比例。

### 7.4 DPO

偏好数据为 $(x,y_w,y_l)$，分别表示 prompt、chosen 和 rejected。定义策略相对 reference 的隐式奖励：

$$
r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}.
$$

DPO 损失：

$$
\mathcal L_{\mathrm{DPO}}=
-\mathbb E_{(x,y_w,y_l)}\left[
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
\right].
$$

直觉：让 chosen 相对 rejected 的**策略-reference 对数概率差**更大。DPO 不需要显式 reward model、critic 和在线 rollout，训练稳定、工程简单；但它依赖离线偏好覆盖，无法主动探索策略更新后出现的新状态。

### 7.5 DAPO

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）是在 GRPO 路线上针对长推理 RL 的一组改进，不是把 DPO 多加一个字母。核心 objective 可概括为：

$$
J_{\mathrm{DAPO}}(\theta)=
\mathbb E\left[
\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left(
r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon_{low},1+\epsilon_{high})\hat A_i
\right)
\right],
$$

其中通常取 $\epsilon_{high}>\epsilon_{low}$。四个关键点：

| 改进 | 机制 | 想解决的问题 |
|---|---|---|
| Clip-Higher | 上界从 $1+\epsilon$ 放宽为 $1+\epsilon_{high}$ | 正 advantage token 的概率增长过早被截断，探索不足 |
| Dynamic Sampling | 过滤组内奖励全 0 或全 1 的 prompt，并继续采样到有效 batch | 全对/全错组的 advantage 为 0，浪费计算 |
| Token-Level Policy Gradient Loss | 用 batch 中所有有效 token 的总数做归一化 | 逐样本平均会让短回答和长回答拥有相同总权重 |
| Overlong Reward Shaping | 接近最大长度时给平滑惩罚，超长截断再给更强惩罚 | 硬截断会把本可完成的长回答突然记为失败，引入奖励噪声 |

一种平滑超长惩罚可写为：

$$
R_{len}(y)=\begin{cases}
0,&|y|\le L_{max}-L_{soft},\\
-\alpha\dfrac{|y|-(L_{max}-L_{soft})}{L_{soft}},
&L_{max}-L_{soft}<|y|\le L_{max},\\
-\alpha,&|y|>L_{max}.
\end{cases}
$$

具体实现的奖励范围和截断规则可以不同，面试重点是说明“把长度边界附近的奖励从突变改为渐变”。

### 7.6 PPO、GRPO、DPO、DAPO 对比

| 方法 | 数据来源 | 是否需要 critic | 是否在线 rollout | 主要 baseline/约束 | 更适合 |
|---|---|---:|---:|---|---|
| PPO | 当前策略生成 | 是 | 是 | GAE + ratio clip + reference KL | 通用 RLHF、需要 token credit 的任务 |
| GRPO | 同 prompt 多采样 | 否 | 是 | 组内相对奖励 + clip | 数学、代码等可验证结果奖励 |
| DPO | 离线 chosen/rejected | 否 | 否 | reference policy 的隐式 KL 约束 | 静态偏好、风格和帮助性对齐 |
| DAPO | 动态筛选后的组采样 | 否 | 是 | 非对称 clip + 有效组 + token 聚合 | 长 CoT、稀疏可验证奖励 |

**一句话选型：**已有高质量静态偏好对时先考虑 DPO；能低成本验证答案、同题多采样时考虑 GRPO/DAPO；轨迹长短差异大、需要学习 value 和细粒度 credit 时 PPO 更自然。最终选择必须把 rollout 吞吐、奖励可靠性和 on/off-policy 偏差一起算进去。

---

## 8. 其他高频公式

### 8.1 MoE

$$
y_t=\sum_{i\in\operatorname{TopK}(g(x_t))}
p_i(x_t)E_i(x_t)+E_{shared}(x_t).
$$

MoE 让每个 token 只激活少量 expert，以较低的 active FLOPs 扩大总参数容量；但不会自动减少总权重显存，并引入路由不均衡、token dispatch/combine 和 All-to-All 通信。

### 8.2 Scaling Law 与训练 FLOPs 粗估

经验 scaling law 常写为：

$$
L(N,D)\approx L_\infty+aN^{-\alpha}+bD^{-\beta},
$$

其中 $N$ 是模型参数量，$D$ 是训练 token 数。dense decoder-only Transformer 的训练计算可粗估：

$$
C\approx6ND.
$$

对 MoE 应更关注 active parameters；注意力、重计算、稀疏路由、视觉塔等会让真实 FLOPs 偏离该估算，最终以 profiler 和有效吞吐为准。

### 8.3 量化

对称均匀量化：

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s),q_{min},q_{max}\right),
\qquad \hat x=sq.
$$

非对称量化加入 zero-point $z$：

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),
\qquad \hat x=s(q-z).
$$

per-channel/group-wise 通常比 per-tensor 更能适应离群值，但 scale/zero-point 元数据和 kernel 更复杂。权重量化主要省权重带宽；KV Cache 量化主要省长上下文解码内存和带宽。

### 8.4 常用评测指标

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F_1=\frac{2PR}{P+R}.
$$

排序任务 DCG/NDCG：

$$
\operatorname{DCG}@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)},
\qquad
\operatorname{NDCG}@K=\frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K}.
$$

面试中不要只报均值：生成式评测还应说明采样次数、pass@k、长度、方差、置信区间、判分器版本和数据污染控制。

---

## 9. 手撕 MHA

这是**面试最低记忆版**，只写 decoder 的 causal self-attention。先背一句口诀：

> **QKV 投影 -> 拆头 -> QK 转置相乘并缩放 -> causal mask -> Softmax 乘 V -> 合头。**

```python
import math
import torch
from torch import nn
import torch.nn.functional as F


def causal_attention(q, k, v):
    # q/k/v: [B, H, T, D]
    T, D = q.shape[-2:]
    score = q @ k.transpose(-2, -1) / math.sqrt(D)

    # 上三角是未来 token，需要屏蔽。
    mask = torch.triu(
        torch.ones(T, T, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    score = score.masked_fill(mask, float("-inf"))
    return score.softmax(dim=-1) @ v


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.D = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # [B, T, C] -> [B, H, T, D]
        q = self.q(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.k(x).view(B, T, self.H, self.D).transpose(1, 2)
        v = self.v(x).view(B, T, self.H, self.D).transpose(1, 2)

        y = causal_attention(q, k, v)

        # [B, H, T, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)
```

面试时一边写一边报两个形状即可：拆头后是 `[B, H, T, D]`，注意力分数是 `[B, H, T, T]`。这里只保留核心计算，额外工程分支不写。

---

## 10. 手撕 GQA

GQA 和上面的 MHA 几乎一样，只记住一个变化：**Q 仍有 `Hq` 个头，K/V 只有 `Hkv` 个头，然后让每组 Q 共享 K/V。**

```python
class GQA(nn.Module):
    def __init__(self, d_model, n_q, n_kv):
        super().__init__()
        assert d_model % n_q == 0
        assert n_q % n_kv == 0
        self.Hq = n_q
        self.Hkv = n_kv
        self.D = d_model // n_q
        self.repeat = n_q // n_kv

        self.q = nn.Linear(d_model, n_q * self.D)
        self.k = nn.Linear(d_model, n_kv * self.D)
        self.v = nn.Linear(d_model, n_kv * self.D)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.Hq, self.D).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hkv, self.D).transpose(1, 2)

        # 每个 KV 头复制给一组 Q 头。
        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)

        y = causal_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)
```

记忆关系：`Hkv = Hq` 是 MHA，`Hkv = 1` 是 MQA，中间情况就是 GQA。

最小形状自测：

```python
x = torch.randn(2, 8, 64)
assert MHA(64, 8)(x).shape == (2, 8, 64)
assert GQA(64, 8, 2)(x).shape == (2, 8, 64)
```

**如果追问 KV Cache：**只缓存复制前的 K/V，形状为 `[B, Hkv, T, D]`，否则会把 GQA 省下的显存重新浪费掉。

---

## 11. 手撕 RoPE、RMSNorm 与 SwiGLU

### 11.1 RoPE

```python
def rotate_half(x):
    # [x0, x1] -> [-x1, x0]
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k):
    # q/k: [B, H, T, D]
    T, D = q.shape[-2:]
    inv_freq = 1 / (
        10000 ** (torch.arange(0, D, 2, device=q.device).float() / D)
    )
    angle = torch.outer(torch.arange(T, device=q.device), inv_freq)
    cos = angle.cos().repeat_interleave(2, dim=-1)
    sin = angle.sin().repeat_interleave(2, dim=-1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

只记一句：**RoPE 对 Q/K 做二维旋转，不处理 V。**

### 11.2 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight
```

### 11.3 SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

---

## 12. 手撕时的检查清单

只检查五件事：

1. 拆头后是不是 `[B, H, T, D]`。
2. K 是否在最后两个维度转置。
3. 是否除以 $\sqrt D$。
4. Softmax 是否沿最后一维。
5. 合头前是否先 `transpose(1, 2).contiguous()`。

实在卡住时，先写出这一行，通常能找回主线：

```python
attention = (q @ k.transpose(-2, -1) / math.sqrt(D)).softmax(-1) @ v
```

---

## 13. 高频追问短答

### 为什么 Attention 用 Softmax，不直接用点积？

Softmax 把分数变成非负、和为 1 的数据相关权重，并通过指数放大相对差异；直接点积没有归一化，输出尺度随序列长度和分数幅度变化。线性注意力可以替换 Softmax，但必须重新设计核函数、归一化与状态更新，精确检索能力也可能变化。

### GQA 为什么能加速解码？

自回归 decode 常受 KV Cache 读取带宽限制。GQA 减少 K/V 头数，使每步从显存读取的历史 K/V 更少；它不等于按同一比例减少全部 FLOPs，实际收益还取决于 batch、序列长度、kernel 和硬件。

### RMSNorm 为什么不减均值也有效？

它保留了对输入尺度的归一化，这是深层残差网络稳定性的关键部分；重中心化并非所有模型都必需。是否更好不是纯理论结论，要看架构、初始化和实测。

### AdamW 为什么比 Adam + L2 更标准？

Adam 的逐元素自适应缩放会同时缩放加到梯度里的 L2 项，导致不同参数获得不同强度的“衰减”；AdamW 直接按比例衰减权重，语义更清晰，也更容易独立调学习率和 decay。

### DPO 和 PPO 最大区别是什么？

DPO 是离线偏好分类式目标，不训练 critic、不在线探索；PPO 用当前策略 rollout，经 reward/value 计算 advantage 后做受约束的在线更新。DPO 工程简单，PPO 能进入新状态但系统成本和不稳定性更高。

### GRPO 真的比 PPO 便宜吗？

它省掉 critic 的参数、前向和优化器状态，但同一 prompt 要生成一组回答。输出很长或组大小很大时，新增 rollout 成本可能超过 critic 成本，所以应比较端到端 token 吞吐，而不是只比较模型数量。

### DAPO 相比 GRPO 改了什么？

不是单一新 loss，而是四项协同：放宽正向概率比上界、只保留有学习信号的动态采样组、按全 batch token 聚合 loss、平滑处理超长回答奖励。它主要解决稀疏结果奖励下的探索、无效 batch、长短样本权重和截断噪声。

---

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

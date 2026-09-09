# 当前时序 2D→3D 算法说明：与 Strided Transformer 的关系

## 结论先行

本项目的第三版模型**采用了论文的核心方法路线**：先对视频每一帧独立运行
RTMDet + RTMPose 得到 2D 关节点；随后只将连续帧的 2D 点序列输入
Transformer，不再读取 RGB 图像；网络先经 VTE 得到全序列 3D 表征，再经 STE
压缩为中心帧表征，并以“全序列 + 中心帧”双监督共同训练。

因此，它是对 Li 等人 *Exploiting Temporal Contexts with Strided Transformer for
3D Human Pose Estimation*（下文称“论文”）的**工程化、缩短时序窗口的实现**，
不是使用论文原始检测器、原始帧数、完整超参数和全部训练细节的逐行复现。

论文链接：[arXiv:2103.14304](https://arxiv.org/pdf/2103.14304)。

## 问题定义与符号

设视频中目标帧为 $t$，当前模型取以它为中心的九帧窗口：

$$
\mathcal{W}_t = \{t-4,t-3,\ldots,t,t+1,\ldots,t+4\}.
$$

对每一帧 $\tau$，2D 检测器输出 17 个 H36M 对齐的关节：

$$
\hat{\mathbf P}^{2D}_{\tau} \in \mathbb{R}^{17\times 2}.
$$

其中 $\hat{}$ 强调这是 **RTMDet + RTMPose 的预测点**，不是 Human3.6M 的人工
2D 真值点。当前版本只使用 $(x,y)$；检测器输出的置信度虽然保存在输入 NPY 中，
但尚未送入网络。

将像素坐标按 H36M 图像尺寸 $W=1000,H=1002$ 归一化：

$$
\tilde{x}=\frac{x-W/2}{W/2},\qquad
\tilde{y}=\frac{y-H/2}{W/2}.
$$

注意 $y$ 也除以 $W/2$，这是当前编码器与 MMPose VideoPoseLifting 的约定，
并非逐轴除以 $H/2$。窗口输入可记作

$$
\mathbf X_t\in\mathbb{R}^{B\times T\times J\times 2}
=\mathbb{R}^{B\times9\times17\times2},
$$

其中 $B$ 是 batch size，$T=9$，$J=17$。MMPose 打包后实际传入 backbone 的布局为
$(B,34,9)$；backbone 会还原为上述含义并转成每帧一个 token。

监督目标是相机坐标系下的 3D 骨架。设骨盆根关节索引 $r=0$，对任意关节 $j$：

$$
\mathbf Y^{rel}_{\tau,j}=\mathbf Y^{cam}_{\tau,j}-\mathbf Y^{cam}_{\tau,r}.
$$

训练时删除恒为零的根关节，只预测其余 16 个关节，所以网络的 3D 输出维度为
$16\times3$。标签单位是米，故验证 `MPJPE=0.060` 对应约 $60\,\mathrm{mm}$。
部署可视化时再使用保存的目标根坐标恢复相机坐标：

$$
\hat{\mathbf Y}^{cam}_{t,j}=\hat{\mathbf Y}^{rel}_{t,j}+\mathbf Y^{cam}_{t,r}.
$$

这一步仅用于与带绝对位置的 H36M 标注对齐；单目真实图片没有真实根位置时，
应以根相对骨架展示结果。

## 当前算法的完整数据流

```text
视频帧 t-4 ... t ... t+4
        │
        ├─ 每帧独立：RTMDet 人体框 + RTMPose-M 17 个 2D 点
        │
        ▼
9 × (17×2) 检测点 + 坐标归一化
        │
        ▼
线性 Pose Embedding (34 → 256) + BatchNorm + ReLU + Dropout
        │ 加 VTE 位置编码
        ▼
VTE：3 个普通 Transformer block（所有 9 帧彼此注意力）
        ├──────────────────────► 全部 9 帧 3D 预测 ──► 序列 MPJPE
        │
        │ 加第一级 STE 位置编码
        ▼
STE-1：注意力 + stride=3 的卷积前馈 / MaxPool，9 → 3
        │
        │ 加第二级 STE 位置编码
        ▼
STE-2：注意力 + stride=3 的卷积前馈 / MaxPool，3 → 1
        ▼
中心帧 t 的 3D 预测 ─────────────────────────────────► 中心帧 MPJPE
```

### 1. 逐帧 2D 检测与时序窗口

二维检测本身没有将前后帧的 RGB 特征融合：每帧可单独运行。时序信息从这里才开始：
同一人的九个连续 2D 姿态组成一个窗口。训练集为 S1/S5/S6/S7/S8，测试集为
S9/S11；输入是本项目预先生成、且与 H36M 3D 标签逐行对齐的 RTMPose 结果。

当前是**非因果（offline）**窗口，预测中心帧时会用到未来四帧。它适合批处理
视频；用于实时相机时，必须改成只使用过去帧并重新训练为 causal 模型，或接受
四帧的显示延迟。

### 2. Pose embedding

每帧的 34 个归一化坐标被展平并映射到 256 维：

$$
\mathbf z^{(0)}_\tau =
\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{BN}(
\mathbf W_e\operatorname{vec}(\tilde{\mathbf P}^{2D}_\tau)+\mathbf b_e)))
+\mathbf e^{VTE}_\tau.
$$

其中 $\mathbf e^{VTE}_\tau$ 是可学习的位置编码。没有位置编码的自注意力会把帧
顺序视为可交换，无法知道“前后关系”。

### 3. VTE：全时域编码器

VTE（Vanilla Transformer Encoder）由 3 个 pre-norm Transformer block 构成。
对第 $l$ 层输入 $\mathbf Z^{(l)}$：

$$
\mathbf A^{(l)}=
\mathbf Z^{(l)}+
\operatorname{MHA}(\operatorname{LN}(\mathbf Z^{(l)})),
$$

$$
\mathbf Z^{(l+1)}=
\mathbf A^{(l)}+
\operatorname{FFN}(\operatorname{LN}(\mathbf A^{(l)})).
$$

MHA 的每个 token 都可以关注九帧中的任意 token。因此若目标帧的手腕被手机遮住、
但相邻帧可见，VTE 有机会依据邻帧姿态补全其语义表征；它不能凭空保证恢复，效果
取决于相邻帧是否真的包含有用信息及训练分布。

VTE 输出仍保留九个时间位置：

$$
\mathbf F_{VTE}\in\mathbb{R}^{B\times256\times9}.
$$

一个 $1\times1$ 卷积回归头将其逐帧映射为完整序列的 3D 预测
$\hat{\mathbf Y}^{VTE}\in\mathbb{R}^{B\times9\times16\times3}$。

### 4. STE：由全序列压缩为一个目标帧

STE（Strided Transformer Encoder）不是简单取 VTE 的中间第 5 帧。它先让所有帧
再次自注意力交互，然后以带步长的卷积前馈网络及 MaxPool 同时下采样残差分支：

$$
\mathbf H=\mathbf Z+\operatorname{MHA}(\operatorname{LN}(\mathbf Z)),
$$

$$
\operatorname{STE}_s(\mathbf Z)=
\operatorname{Pool}_s(\mathbf H)+
\operatorname{ConvFFN}_{s}(\operatorname{LN}(\mathbf H)).
$$

当前两级 $s=3$：

$$
9\xrightarrow[]{s=3}3\xrightarrow[]{s=3}1.
$$

最后的单一 token 是从整个窗口聚合出的“中心帧任务”特征，而不应解释为某个原始
帧被机械选中。第二个 $1\times1$ 回归头给出
$\hat{\mathbf Y}^{STE}_t\in\mathbb{R}^{B\times16\times3}$。

### 5. 双监督（full-to-single loss）

训练同时约束 VTE 的全序列预测与 STE 的中心帧预测：

$$
\mathcal L=
\mathcal L_{single}+\lambda\mathcal L_{sequence},\qquad\lambda=1.0.
$$

本实现两个项均为带可见性权重的 MPJPE：

$$
\operatorname{MPJPE}(\hat{\mathbf Y},\mathbf Y)=
\frac{1}{\sum w_{i,j}}\sum_{i,j}w_{i,j}
\left\|\hat{\mathbf Y}_{i,j}-\mathbf Y_{i,j}\right\|_2.
$$

其中 $i$ 表示帧或样本，$j$ 表示关节，$w$ 来自 3D 标注的可见性。全序列损失迫使
VTE 的每个时间位置都有可靠的 3D 几何含义，避免 STE 只能学习一个黑盒式压缩；
中心损失则直接优化最终部署所需的中心帧输出。这就是“full-to-single”双监督，
不是两个相互独立模型的集成。

## 与论文的相同点

| 维度 | 论文 | 当前实现 |
|---|---|---|
| 输入范式 | 连续帧 2D 关节点，不直接用 RGB 时序特征 | 连续帧 RTMPose 2D 关节点，不直接用 RGB 时序特征 |
| 第一阶段 | VTE 保留全序列时序表征 | 3 个 VTE block，保留 9 帧表征 |
| 第二阶段 | STE 以步长逐级压缩序列到单帧 | 两级 stride=3，`9→3→1` |
| 训练目标 | 全序列输出和单帧输出联合监督 | 序列 MPJPE + 中心帧 MPJPE，权重均为 1 |
| 任务 | 单目视频 2D-to-3D pose lifting | 单目视频 2D-to-3D pose lifting |
| 推理语义 | 从邻帧姿态上下文改善目标帧 3D | 同样利用前后四帧改善中心帧 3D |

所以用户所说“先用没有时序信息的图片推 2D 点，再用 2D 点的时序信息进入 VTE、STE”
是正确的：这正是两者最关键的共同点。

## 与论文的关键区别

| 维度 | 论文路线 | 本项目当前实现 | 影响 |
|---|---|---|---|
| 时序长度 | 常用 27 帧，示例压缩 `27→9→3→1` | 固定 9 帧，`9→3→1` | 当前上下文更短、成本更低，但无法直接声称达到论文长窗口收益。 |
| 帧率与时间跨度 | 论文实验设置使用其指定的采样/检测数据 | 本项目使用 H36M `fps10` | 9 帧覆盖约 0.8 秒；时序分辨率与论文不等价。 |
| 2D 前端 | 论文使用其论文实验中的 2D 输入/检测设定 | RTMDet + RTMPose-M，转为 H36M17 | 更贴近本项目部署链路，但检测噪声分布不同。 |
| 模型超参数 | 论文的层数、宽度、dropout、实现细节以原文为准 | 256 维、8 头、FFN 512、3 层 VTE、dropout 0.25 | 是可运行的复现性配置，不是原文的逐项复刻。 |
| STE 实现 | 论文定义的 Strided Transformer 结构 | PyTorch `MultiheadAttention` + Conv1d CFFN + MaxPool 残差 | 保留“注意力后逐级 stride 压缩”的算法本质，但算子组织不保证逐层参数同构。 |
| 3D 表示 | 以论文数据处理流程为准 | 骨盆根相对、删除根后预测 16 点，单位米 | 适合现有 MMPose codec 和 MPJPE；绝对位置不是网络学习目标。 |
| 置信度 | 取决于论文的 2D 输入设置 | 当前丢弃 RTMPose confidence，仅输入 $(x,y)$ | 遮挡时没有明确告知模型哪些点不可信。 |
| 数据加载 | 与论文实现无关 | 自定义惰性数据集 | 这是 WSL 内存优化，不改变网络数学定义。 |
| 因果性 | 论文可按其评测设置使用邻帧 | 当前明确为非因果中心窗口 | 离线视频有效；实时需延迟或重训 causal 版本。 |

## 当前配置与代码对应

| 项目 | 当前值 | 代码位置 |
|---|---:|---|
| 输入关节 / 通道 | 17 / 34（仅 x,y） | `strided_transformer_h36m_rtmpose_9frm.py` |
| 序列长度 | 9 | 同上 |
| VTE 层数 | 3 | 同上 |
| 嵌入维度 / 注意力头 | 256 / 8 | 同上 |
| STE 步长 | `(3, 3)` | 同上 |
| 3D 输出关节 | 16 个非根关节 | `codecs.py`、`full_to_single_head.py` |
| 优化器 | AdamW，$lr=10^{-3}$，weight decay $10^{-4}$ | 训练配置 |
| 学习率衰减 | epoch 55、70 各乘 0.1 | 训练配置 |
| batch / workers | 256 / 4 | 训练配置 |
| 验证指标 | MPJPE、P-MPJPE | 训练配置 |

`LazyHuman36mDataset` 将 30 多万个九帧窗口按需组装，检测点 NPY 以 memory-map
读取。这只减少 Python 对象复制和 WSL 内存占用；每次送入模型的样本、标签和损失
与非惰性数据集的定义相同。

## 指标应如何解释

- **MPJPE**：根对齐之后的逐关节欧氏距离平均值；`0.060 m = 60 mm`。它包含姿态、
  尺度和朝向方面的误差。
- **P-MPJPE**：先对预测和标签作刚体 Procrustes 对齐再计算，因此移除了全局
  旋转、平移和尺度差异。它更接近“骨架相对形状是否正确”，但不能取代 MPJPE。
- 训练日志中的 `mpjpe` 是当前训练 batch 的目标项，不能与完整 S9/S11 验证集的
  `MPJPE` 直接比较，也不能单独用来判断泛化能力。

## 为什么当前时序提升可能有限

时序模型不是必然显著优于单帧模型，当前实验存在几个具体限制：

1. H36M 的动作与相机相对受控，单帧 2D 点已经能恢复大部分姿态；可由邻帧补足的
   信息有限。
2. 相邻帧高度相似，九帧容易带来重复信息，而非新增的视角信息。
3. RTMPose 的检测误差可能跨连续帧相关，例如持续将一只手定位偏移；时间注意力
   不能自动消除这种系统误差。
4. 当前仅输入 $(x,y)$，没有把低置信度、缺失或遮挡显式编码为信号。
5. 本项目单帧 TCN 与当前 Transformer 在结构、编码和损失上并不完全一致；比较
   两者不是严格的“仅增加时序”消融实验。

要验证纯时序收益，应在**完全相同的 detector、codec、Transformer 宽度、损失和
训练计划**下，额外训练 `seq_len=1` 的基线，再与 `seq_len=9` 对比。要验证遮挡
恢复，应对同一测试集人为遮挡/置零部分关节，并报告按遮挡强度分组的 MPJPE。

## 可以继续演进但尚未实现的方向

1. **置信度感知输入**：将每帧输入扩展为 $(x,y,c)$，即 $17\times3=51$ 通道；
   $c$ 为 RTMPose 检测置信度。
2. **遮挡掩码增强**：再加入二值 mask，输入成为 $(x,y,c,m)$；训练时随机对连续
   1/2/4 帧的部分关节注入缺失或噪声，但 3D 标签不变，使模型学习借邻帧恢复。
3. **因果版本**：窗口改为 $\{t-8,\ldots,t\}$，重新训练后用于实时场景；它避免
   未来帧依赖，但通常比非因果模型精度低。
4. **长窗口消融**：在显存和训练时间允许时，试验 27 帧、三级 stride=3；必须用
   同一数据采样和独立实验目录，不能覆盖当前 9 帧结果。
5. **真实遮挡外部验证**：H36M 可完成第一版训练；若要证明真实复杂遮挡下的泛化，
   需另选带 3D 标注的外部数据集并明确其许可证和关节定义映射。

上述任一项都会改变输入分布或网络参数，均应作为新的 v4 实验训练，保留当前 v3
checkpoint 作为可比较基线。

## 复现边界

本说明描述的是仓库 `projects/strided_transformer_pose_lift/` 的实际实现，而不是对
论文代码的宣称性复刻。报告中宜表述为：

> 我们基于 Strided Transformer 的 VTE–STE full-to-single 思想，使用 RTMDet +
> RTMPose 生成的 H36M17 2D 检测点，构建了一个 9 帧、非因果的时序 2D-to-3D
> pose lifting 模型；其窗口长度、2D 前端、数据采样、网络超参数和工程实现与原论文
> 存在差异。

# v4 审核稿：置信度感知与合成遮挡增强的时序 2D→3D Pose Lifting

## 目标

v3 仅接收每个关节的 $(x,y)$。当 RTMPose 因遮挡给出低置信度或错误位置时，模型无法
明确区分“真实位置”和“可靠性很低的猜测”。v4 的目标不是宣称解决任意真实遮挡，而是：

1. 让输入显式携带检测置信度；
2. 训练模型在连续帧的局部肢体 2D 证据缺失时，仍利用可见关节和相邻帧恢复中心帧 3D；
3. 在干净和受控合成遮挡测试集上分别报告性能。

它是独立实验，不会覆盖 v3 的配置、checkpoint 或日志。

## 输入与监督

每帧、每关节特征从 v3 的：

$$
\mathbf q_{t,j}^{v3}=(x_{t,j},y_{t,j})
$$

扩展为：

$$
\mathbf q_{t,j}^{v4}=(x_{t,j},y_{t,j},c_{t,j},m_{t,j}),
$$

其中：

- $c_{t,j}$：RTMPose 检测置信度；
- $m_{t,j}\in\{0,1\}$：输入遮挡掩码，1 表示原始输入证据保留，0 表示人为移除；
- 九帧窗口共有 $17\times4=68$ 个输入通道。

三维标签不随增强改变，仍为 H36M 根相对 16 个非根关节：

$$
\mathbf Y^{rel}_{t,j}=\mathbf Y^{cam}_{t,j}-\mathbf Y^{cam}_{t,0}.
$$

因此训练目标是：即使 $\tilde{\mathbf Q}$ 的部分观测被破坏，仍最小化：

$$
\mathcal L=
\operatorname{MPJPE}(f(\tilde{\mathbf Q}),\mathbf Y_t)
+\operatorname{MPJPE}(f_{seq}(\tilde{\mathbf Q}),\mathbf Y_{t-4:t+4}).
$$

## 合成遮挡机制

训练集中每个九帧窗口以 0.75 概率执行一次增强：

1. 随机选择一个解剖关节组：左/右腿、左/右臂，或躯干/头部；
2. 随机选择连续 1、2 或 4 帧；
3. 将该时间段、该组关节的 $(x,y)$ 填零，$c$ 置零，$m$ 置零；
4. 保持原始 3D 目标和 3D 可见性权重不变。

这意味着零坐标本身不应被网络理解为有效位置，因为 $c=0,m=0$ 明确表明它没有视觉
证据。模型必须借助邻帧和其他关节完成预测。

验证集默认不加合成遮挡，所有 $m=1$，因此可监控“遮挡鲁棒性是否以干净样本精度为代价”。

## 当前代码和配置

| 路径 | 内容 |
|---|---|
| `transforms.py` | `TemporalJointOcclusion` 输入增强。 |
| `codecs.py` | 支持拼接 confidence 和 mask。 |
| `models/strided_transformer.py` | 支持每关节 2 或 4 维输入。 |
| `strided_transformer_h36m_rtmpose_occconf_9frm.py` | v4 独立训练配置。 |
| `work_dirs/strided_transformer_h36m_rtmpose_occconf_9frm/` | 未来训练输出目录。 |

最小检查已验证：遮挡不修改 3D 标签，编码输出 `(68, 9)`，模型输出 `(1,16,3)`。

## 训练参数与成本

除输入维度和输入增强外，v4 保持 v3 主要参数不变：9 帧、3 VTE 层、256 维、8 头、
`9→3→1` STE、batch 256、4 workers、AMP、AdamW、80 epoch 和 epoch 55/70 学习率衰减。

输入 embedding 从 34 到 68 通道只带来极小的计算增加；训练时间预计仍约 15–17 小时。
由于输入层维度变化，不可直接 `--resume` v3 checkpoint；v3 保留为对照。

## 审核通过后训练命令

在 `src/` 目录执行：

```bash
PYTHONPATH="$PWD" python3 tools/train.py \
  projects/strided_transformer_pose_lift/strided_transformer_h36m_rtmpose_occconf_9frm.py \
  --work-dir work_dirs/strided_transformer_h36m_rtmpose_occconf_9frm \
  --amp
```

## 训练完成后必须补做的评测

仅报告干净 H36M MPJPE 不足以证明遮挡增强有效。至少需要对 S9/S11 的同一检测点测试集
生成下列受控输入，并分别比较 v3 与 v4：

| 组别 | 破坏方式 | 关键指标 |
|---|---|---|
| Clean | 无破坏 | MPJPE、P-MPJPE，检查是否损失常规精度。 |
| Joint-1 | 单关节遮挡 1 帧 | 中心帧 MPJPE。 |
| Limb-2 | 单肢体遮挡连续 2 帧 | 中心帧 MPJPE、被遮挡关节 MPJPE。 |
| Limb-4 | 单肢体遮挡连续 4 帧 | 同上，验证邻帧恢复极限。 |

若 v4 仅在 clean 集下降、却没有降低 `Limb-2/Limb-4` 的被遮挡关节误差，则该路线不成立；
若 v4 在遮挡组有明显改善且 clean 损失可接受，才能称为遮挡鲁棒性提升。

## 仍然存在的边界

- 合成遮挡不是现实遮挡：它未模拟服装、多人交叠、人体检测框丢失和身份切换；
- 当前 v4 仍是非因果 9 帧模型，离线视频适用，实时场景需要因果重训或四帧延迟；
- 多人视频尚无稳定 tracking，第一版部署仍限制时序模型为最高置信度单人；
- H36M 的受控环境不能替代外部真实遮挡数据集评测。

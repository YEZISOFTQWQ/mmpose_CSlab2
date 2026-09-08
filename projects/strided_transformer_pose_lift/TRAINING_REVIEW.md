# Strided Transformer 训练审核稿

## 目标与边界

本实验将当前 RTMDet + RTMPose M 所产生的连续 2D H36M17 关键点，提升为中心帧的根相对 3D 姿态。它是第二版单帧 TCN 模型的独立第三条路线，不会修改或覆盖：

`work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/best_MPJPE_epoch_75.pth`

训练输出只会写入 `work_dirs/strided_transformer_h36m_rtmpose_9frm/`。本文件审核通过前，不执行训练命令。

## 方法

实现参考 Li 等人的 [Strided Transformer](https://arxiv.org/pdf/2103.14304)：

```text
9 帧 RTMPose 2D 点 (17×2)
        │ 按 H36M 图像宽高归一化
        ▼
pose embedding → VTE: 3 层全时域自注意力 ──→ 全部 9 帧 3D 预测
                        │                           │
                        └→ STE: 两层 [注意力 + stride=3 卷积] → 中心帧 3D 预测
                                                           │
                 loss = MPJPE(9 帧 VTE) + MPJPE(中心帧 STE)
```

- **VTE**（Vanilla Transformer Encoder）先让每个帧 token 看见整个窗口的姿态上下文。
- **STE**（Strided Transformer Encoder）以两次 stride=3，将 `9 → 3 → 1`；输出天然对应居中的第 5 帧。
- 3D 坐标均以骨盆为原点，移除骨盆后学习余下 16 点，单位为 H36M 标注的米。因此日志中 `MPJPE=0.06` 就是约 60 mm。
- 当前用 9 帧、3 个 VTE 层、256 维嵌入、8 头、512 维前馈层。论文的典型设置是 27 帧和三级步长压缩；这里采用 H36M fps10 文件，9 帧可先验证时序收益并控制显存与训练时间。

## 数据与一致性

所需的文件已经在本机数据路线中：

```text
data/h36m/annotation_body3d/fps10/
├── h36m_train_keypoints.npz                # S1/S5/S6/S7/S8 的 2D/3D 标签
├── h36m_test_keypoints.npz                 # S9/S11 的保留测试集
├── rtmdet_rtmpose_m_train_h36m17.npy       # 训练输入的真实检测点
└── rtmdet_rtmpose_m_test_h36m17.npy        # 测试输入的真实检测点
```

这里**没有**依赖缺失的 `cameras.pkl`。2D 点的归一化规则是
`((x, y) - (500, 501)) / 500`，即 H36M 图像 1000×1002 的宽高归一化；后续视频推理也必须按输入帧实际宽高采用同一规则。训练和部署输入都是 RTMDet + RTMPose 检测结果，避免以 GT 2D 训练、检测点部署的分布落差。

初版不加入水平翻转增强，因为它必须同步翻转整段 2D 点、整段 3D 标签和左右关节索引；在这个新双尺度标签路径上，先保证标签正确，再作为可复现实验单独加入。

## 审核前的只读检查

在 `src` 目录执行：

```bash
python3 projects/strided_transformer_pose_lift/tools/validate_temporal_inputs.py
```

它只读取文件，检查 NPZ 字段、检测数组形状、NaN/Inf，以及按视频切分后的有效 9 帧窗口数量。

## 审核通过后才执行

```bash
PYTHONPATH="$PWD" python3 tools/train.py \
  projects/strided_transformer_pose_lift/strided_transformer_h36m_rtmpose_9frm.py \
  --work-dir work_dirs/strided_transformer_h36m_rtmpose_9frm
```

首轮采用 batch size 256、4 个数据加载进程、80 epoch，在第 55、70 epoch 将学习率 `1e-3` 分别降为原来的 0.1。首个 epoch 的显存若超出 GPU 容量，只将配置中 `batch_size=256` 降到 128；模型和学习率日程保持不变。训练完成后，以 `best_MPJPE_epoch_*.pth` 作为唯一候选，与 v2 的 S9/S11 检测点测试结果比较。

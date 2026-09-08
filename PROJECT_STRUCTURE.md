# 项目文件结构

本仓库以 MMPose 源码为基础；MMPose 的核心源码与官方配置没有被修改。
本项目新增的内容集中在 `projects/single_image_pose_lift/`、
`projects/strided_transformer_pose_lift/`、`data/`、`work_dirs/` 与
`artifacts/`。

```text
src/
├── configs/                         # MMPose 官方模型配置
├── demo/                            # MMPose 官方 demo 与检测器配置
├── mmpose/                          # MMPose 核心源码（未修改）
├── tools/                           # 工具脚本
│   ├── dataset_converters/
│   │   └── convert_h36m_annot_h5.py # 新增：HDF5 注释转训练 NPZ
│   └── render_pose3d_views.py        # 新增：多视角 3D 骨架渲染
├── projects/
│   ├── single_image_pose_lift/      # 新增：单图 2D→3D 模型、数据与评估代码
│   └── strided_transformer_pose_lift/ # 新增：时序 Strided Transformer 实验
├── data/
│   ├── h36m/                        # H36M 注释与训练输入
│   └── h36m_raw/archives/           # S1/S5/S6/S7/S8/S9/S11 原始 tar 包
├── work_dirs/                       # 两轮训练的最佳模型、日志和指标
├── artifacts/                       # 权重、输入媒体、最终视频与报告图
└── PROJECT_STRUCTURE.md             # 本文档
```

## `projects/single_image_pose_lift/`

单图 2D→3D pose lifting 项目代码。

| 文件 | 用途 |
|---|---|
| `image_pose_lift_tcn_h36m_keypoints.py` | 第一版：输入 H36M 真值 2D 点的 TCN 训练配置。 |
| `image_pose_lift_tcn_h36m_rtmpose_v2.py` | 第二版：输入 RTMDet + RTMPose 检测点的 TCN 训练配置。 |
| `generate_h36m_rtmpose_from_tar.py` | 从原始 subject tar 流式读取图片，批量生成对齐的 H36M 17 点检测数组。 |
| `benchmark_s1_rtmpose_lift.py` | 图像 → 2D 点 → 3D 预测 → 真实 3D 的批量评估与作图。 |
| `demo_h36m_keypoints_lift.py` | 使用指定 2D 点测试 3D lifter 并可视化。 |
| `demo_h36m_rtmpose_lift.py` | RTMDet + RTMPose + 3D lifter 的单图端到端 demo。 |
| `README.md` | 数据格式、训练路线与限制说明。 |

## `projects/strided_transformer_pose_lift/`

第三条、尚未开始训练的时序 2D→3D 路线。以连续 9 帧 RTMPose 17 点为输入，
使用论文 *Exploiting Temporal Contexts with Strided Transformer for 3D
Human Pose Estimation* 的 VTE + STE 全序列到中心帧监督思路。

| 文件 | 用途 |
|---|---|
| `strided_transformer_h36m_rtmpose_9frm.py` | 独立训练配置；结果写入新的 `work_dirs/strided_transformer_h36m_rtmpose_9frm/`。 |
| `codecs.py` | 9 帧 2D 归一化，以及中心帧和整段根相对 3D 标签编码。 |
| `lazy_h36m_dataset.py` | 惰性组装时序窗口的数据集，避免 9 帧样本在启动时被整体复制到内存。 |
| `models/strided_transformer.py` | VTE + 逐级 stride=3 的 STE 骨干网络。 |
| `models/full_to_single_head.py` | 全序列和中心帧的双 MPJPE 损失头。 |
| `tools/validate_temporal_inputs.py` | 训练前只读检查数据文件、形状、数值与可用窗口数。 |
| `tools/benchmark_batch_size.py` | 不保存 checkpoint 的训练吞吐基准，用于选择 batch size。 |
| `TRAINING_REVIEW.md` | 方法、参数、数据要求和待审核训练命令。 |

## `data/`

### `data/h36m/`

| 路径 | 内容 |
|---|---|
| `h36m_annot.tar` | H36M 配对注释原始压缩包。 |
| `annotation_body3d/fps10/h36m_train_keypoints.npz` | S1/S5/S6/S7/S8 的 312,188 个训练样本。 |
| `annotation_body3d/fps10/h36m_test_keypoints.npz` | S9/S11 的 109,867 个 held-out 测试样本。 |
| `annotation_body3d/fps10/rtmdet_rtmpose_m_train_h36m17.npy` | 与训练集逐行对齐的 RTMDet + RTMPose 17 点输入。 |
| `annotation_body3d/fps10/rtmdet_rtmpose_m_test_h36m17.npy` | 与测试集逐行对齐的 RTMDet + RTMPose 17 点输入。 |
| 相同文件名后的 `.progress.json` | 2D 点生成进度与 subject 完成状态。 |
| 相同文件名后的 `.fallbacks.csv` | RTMDet 未检测到人体时使用 H36M 框回退的记录。 |

### `data/h36m_raw/archives/`

保留的原始 Human3.6M subject 包：`S1.tar`、`S5.tar`、`S6.tar`、`S7.tar`、
`S8.tar`、`S9.tar`、`S11.tar`。生成 2D 点时按需从 tar 读取图片，不解压完整 RGB 数据集。

## `work_dirs/`

| 目录 | 最佳模型 | 含义 |
|---|---|---|
| `image_pose_lift_tcn_h36m_keypoints/` | `best_MPJPE_epoch_75.pth` | 第一版：真值 2D 输入基线。 |
| `image_pose_lift_tcn_h36m_rtmpose_v2/` | `best_MPJPE_epoch_75.pth` | 第二版：RTMPose 输入模型；部署推荐使用。 |
| `strided_transformer_h36m_rtmpose_9frm/` | 尚未产生 | 第三版：9 帧 RTMPose 输入的 Strided Transformer；训练审核通过后才创建。 |

每个目录下的日期子目录保留了 MMEngine 训练日志和 `vis_data/` 指标 JSON。
非最佳 checkpoint 已清理。

## `artifacts/`

| 目录 | 内容 |
|---|---|
| `input/` | 用户提供的原始图片、视频及其单图推理结果。 |
| `models/` | 当前复现所需的 `rtmdet_m`、`rtmpose_m`、`motionbert` 权重。 |
| `model_archives/gt2d_tcn_epoch75/` | 第一版最佳模型与训练配置的额外归档副本。 |
| `results/official_video/` | 官方 MotionBERT 对输入视频的 3D 推理视频及逐帧 JSON。 |
| `results/report_figures/` | 报告保留图：第一版单图、RTMPose 输入模型，以及 S9/S11 动作中段对比与排名。 |

## 常用模型路径

```text
# 第一版：真值 2D 输入
work_dirs/image_pose_lift_tcn_h36m_keypoints/best_MPJPE_epoch_75.pth

# 第二版：RTMDet + RTMPose 输入（推荐）
work_dirs/image_pose_lift_tcn_h36m_rtmpose_v2/best_MPJPE_epoch_75.pth
```

## 版本控制说明

数据集、权重、训练输出与可视化结果通常不应提交到 Git；它们均为本地实验资产。
项目代码和说明文件位于 `projects/single_image_pose_lift/`、`tools/` 与本文档中。

## 所有成员提交规则

以下规则适用于本仓库的所有协作者和自动化代理。

### 可以提交

- `projects/single_image_pose_lift/`、`projects/strided_transformer_pose_lift/`
  中的源代码、训练配置、评估脚本和说明文档；
- `tools/` 中新增或修改的可复现数据转换、评估、渲染脚本；
- 小型文本配置、Markdown 文档、依赖说明和 `.gitignore` 规则；
- 不含模型参数、隐私内容或受限数据的测试代码。

### 禁止提交

- `data/` 下的 Human3.6M 原始包、注释、NPZ、NPY、进度文件及其他数据集；
- `artifacts/` 下的模型权重、输入图片/视频、推理 JSON、可视化图片和临时缓存；
- `work_dirs/` 下的 checkpoint、日志、指标 JSON、TensorBoard/可视化数据；
- 任何 `.pth`、`.pt`、`.ckpt`、`.tar`、`.tgz`、大型 `.npy`、`.npz`、视频或图片文件；
- API Key、访问令牌、个人路径、隐私图像或受许可证限制而不能再分发的内容；
- 未经明确讨论的 MMPose 核心源码、官方配置或第三方依赖的直接改动。

### 提交前检查

1. 运行 `git status --short`，确认暂存区只包含本次任务相关的源代码和文档。
2. 运行 `git diff --check`，修复空白字符错误。
3. 检查新增文件大小；若文件不是文本或明显超过合理的代码/文档体积，不应提交。
4. 对训练或推理脚本，至少执行语法检查或最小可复现运行，并在提交说明中写明验证方式。
5. 不提交他人已有的未提交改动；遇到不确定归属的文件，应先询问维护者。

### 提交粒度与命名

- 一次提交只完成一个明确目标，例如“新增 tar 流式 2D 点生成器”或“修正 pose-lifter 可视化坐标系”。
- 提交信息使用简短祈使句，可采用：`feat: ...`、`fix: ...`、`docs: ...`、`test: ...`。
- 训练结果应写入实验记录或文档，记录模型配置、checkpoint 路径、数据版本和 MPJPE；不要把结果文件本身提交到 Git。

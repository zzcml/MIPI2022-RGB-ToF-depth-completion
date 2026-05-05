# MIPI2022 RGB+ToF depth completion track

## 伪标签训练功能 (Pseudo-Label Training)

本项目扩展了三个先进的深度估计模型作为教师模型，用于生成伪标签训练学生模型：

### 1. Lotus-2 伪标签训练
基于 [Lotus-2](https://github.com/EnVision-Research/Lotus-2.git) 作为教师模型。

**模块位置**: `models/lotus/`

**快速开始**:
```bash
# 第一步：验证模块
python verify_all_modules.py

# 第二步：运行训练示例
python models/lotus/train_pseudo_label.py \
    --teacher_task_name "depth" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./lotus_output" \
    --num_epochs 10 \
    --batch_size 8
```

**核心功能**:
- 使用 Lotus-2 生成高质量伪标签
- 支持预计算或在线生成伪标签
- 提供 L1、置信度加权损失函数
- 完整的训练、验证和检查点管理

**注意**: Lotus-2 需要安装 `diffusers` 和 `transformers` 才能进行实际推理。验证脚本会检测这些依赖并给出提示。

### 2. Depth Anything V2 伪标签训练
基于 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)，使用 ViT-L 作为教师，ViT-S/ViT-B 作为学生。

**模块位置**: `models/da2/`

**快速开始**:
```bash
# 第一步：验证模块
python verify_all_modules.py

# 第二步：使用 ViT-S 作为学生模型
python models/da2/train_pseudo_label.py \
    --student_model_type "vits" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./da2_output" \
    --num_epochs 20 \
    --batch_size 4

# 使用 ViT-B 作为学生模型
python models/da2/train_pseudo_label.py \
    --student_model_type "vitb" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./da2_output" \
    --num_epochs 20
```

**支持的模型组合**:
- 教师: `vitl` (固定，Depth Anything V2 Large)
- 学生: `vits` (Small), `vitb` (Base)

**核心功能**:
- ViT-L 教师生成伪标签
- 知识蒸馏到轻量级学生模型
- SILog 损失函数优化深度估计
- TensorBoard 日志支持
- 官方源码集成 (`depth_anything_v2/` 目录)

### 3. RoboDepth Zoo 伪标签训练
基于 [RoboDepth](https://github.com/worldbench/RoboDepth) 模型库，支持 24+ 种架构。

**模块位置**: `models/zoo/`

**快速开始**:
```bash
# 第一步：验证模块
python verify_all_modules.py

# 第二步：使用 MonoDepth2 教师训练 ResNet-50 学生
python models/zoo/train_pseudo_label.py \
    --teacher_model "monodepth2_resnet18" \
    --student_model "monodepth2_resnet50" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./zoo_output" \
    --max_epochs 15

# 使用 MonoViT 教师训练 MonoDepth2 学生
python models/zoo/train_pseudo_label.py \
    --teacher_model "monovit_mpvit" \
    --student_model "monodepth2_resnet50" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./zoo_output"
```

**支持的模型家族** (已下载源码):
- `monodepth2_*` - MonoDepth2 (ResNet18/50/101)
- `monovit_*` - MonoViT (MPViT)
- `diffnet_*` - DIFFNet (HRNet + Attention)
- `litemono_*` - Lite-Mono (Lightweight)
- `dynadepth_*` - DynaDepth (Dynamic)
- `radepth_*` - RA-Depth (Recurrent Attention)

**完整模型列表** (24+ 模型):
- ResNet 系列：`monodepth2_resnet18/50/101`
- DINOv2 系列：`robodepth_dinov2_base/large`
- ViT 系列：`robodepth_vit_small/base`
- 以及其他先进模型：DIFFNet, Lite-Mono, DynaDepth, RA-Depth 等

**核心功能**:
- 灵活的教师 - 学生模型组合 (任意 zoo 模型)
- BerHu、MSE、SILog 多种损失函数
- 混合精度训练支持
- 自动学习率调度
- 官方源码集成 (`models/` 目录包含真实模型实现)

---

## 通用训练参数

所有训练脚本支持以下通用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 未标注图像目录 | 必需 |
| `--output_dir` | 输出目录 | `./output` |
| `--num_epochs` / `--max_epochs` | 训练轮数 | 10 |
| `--batch_size` | 批次大小 | 4 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--image_size` | 输入图像尺寸 | 518 |
| `--precomputed_labels` | 是否使用预计算伪标签 | False |
| `--labels_dir` | 预计算标签目录 | None |
| `--loss_type` | 损失函数类型 | 各模块默认 |
| `--device` | 计算设备 | `cuda` |

---

## 验证所有模块

运行综合验证脚本确认所有模块正常工作：

```bash
python verify_all_modules.py
```

**预期输出**:
```
============================================================
 伪标签训练模块综合验证系统
 验证内容：Lotus-2, Depth Anything V2, RoboDepth Zoo
============================================================

验证 Lotus-2 模块...
✓ 源代码目录存在
✓ 成功导入 Lotus2Teacher 和配置类
✓ 损失函数计算正常

验证 Depth Anything V2 模块...
✓ 源代码目录存在
✓ StudentModelWrapper (ViT-S) 实例化成功
✓ 模型结构已初始化

验证 RoboDepth Zoo 模块...
✓ 模型源码目录存在
✓ 模型 [monodepth2_resnet18] 加载并推理成功
✓ 模型 [monovit_mpvit] 加载并推理成功

============================================================
🎉 所有模块验证通过！可以开始使用。
============================================================
```

---

## 传统训练与测试

### Training
To run the training of the baseline:

```bash
sh train.sh (save_path)
```

### Testing
To run the testing of the baseline:

```bash
sh test.sh (test_data_dir) (trained_model) (save_name) (data_list)
```

We provide a trained model of the baseline in [baseline.pt](https://drive.google.com/file/d/1DL6JcYaDSKYph3T93RFDUVIPzQvKFw1n/view?usp=sharing)

---

## 环境要求

**无需安装额外 pip 环境**即可使用核心功能。模块设计为：
- 优雅处理缺失的依赖（如 `diffusers`、`torchvision`）
- 提供清晰的错误提示
- 支持纯 PyTorch 实现的核心功能

如需完整功能（如实际推理），请根据对应模型的官方文档安装依赖：

- **Lotus-2**: `pip install diffusers transformers peft`
- **Depth Anything V2**: `pip install timm opencv-python`
- **RoboDepth Zoo**: `pip install timm einops`

---

## 项目结构

```
/workspace
├── verify_all_modules.py          # 综合验证脚本
├── README.md                       # 本文件
└── models/
    ├── lotus/                      # Lotus-2 伪标签训练模块
    │   ├── lotus_src/              # Lotus-2 官方源码
    │   ├── __init__.py
    │   ├── config.py               # 配置类
    │   ├── teacher_model.py        # 教师模型包装器
    │   ├── pseudo_label_dataset.py # 数据集类
    │   ├── train_utils.py          # 训练工具
    │   └── train_pseudo_label.py   # 训练脚本
    │
    ├── da2/                        # Depth Anything V2 伪标签训练模块
    │   ├── depth_anything_v2/      # DA2 官方源码
    │   ├── __init__.py
    │   ├── config.py
    │   ├── teacher_model.py
    │   ├── student_model.py
    │   ├── pseudo_label_dataset.py
    │   ├── train_utils.py
    │   └── train_pseudo_label.py
    │
    └── zoo/                        # RoboDepth Zoo 伪标签训练模块
        ├── models/                 # 24+ 个真实模型源码
        │   ├── monodepth2/
        │   ├── monovit/
        │   ├── diffnet/
        │   └── ...
        ├── __init__.py
        ├── config.py
        ├── model_wrapper.py
        ├── pseudo_label_dataset.py
        ├── train_utils.py
        └── train_pseudo_label.py
```

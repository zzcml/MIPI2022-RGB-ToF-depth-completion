# MIPI2022 RGB+ToF depth completion track

## 伪标签训练功能 (Pseudo-Label Training)

本项目扩展了三个先进的深度估计模型作为教师模型，用于生成伪标签训练学生模型：

### 1. Lotus-2 伪标签训练
基于 [Lotus-2](https://github.com/EnVision-Research/Lotus-2.git) 作为教师模型。

**模块位置**: `models/lotus/`

**快速开始**:
```bash
# 验证模块
python verify_all_modules.py

# 运行训练示例
python models/lotus/train_pseudo_label.py \
    --teacher_task_name "depth_estimation" \
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

### 2. Depth Anything V2 伪标签训练
基于 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)，使用 ViT-L 作为教师，ViT-S/ViT-B 作为学生。

**模块位置**: `models/da2/`

**快速开始**:
```bash
# 使用 ViT-S 作为学生模型
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
- 教师: `vitl` (固定)
- 学生: `vits` (Small), `vitb` (Base)

**核心功能**:
- ViT-L 教师生成伪标签
- 知识蒸馏到轻量级学生模型
- SILog 损失函数优化深度估计
- TensorBoard 日志支持

### 3. RoboDepth Zoo 伪标签训练
基于 [RoboDepth](https://github.com/worldbench/RoboDepth) 模型库，支持多种架构。

**模块位置**: `models/zoo/`

**快速开始**:
```bash
# 使用 DINOv2-Large 教师训练 ResNet-50 学生
python models/zoo/train_pseudo_label.py \
    --teacher_model_type "robodepth_dinov2_large" \
    --student_model_type "robodepth_resnet50" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./zoo_output" \
    --num_epochs 15

# 使用 DINOv2-Base 教师训练 ViT-Base 学生
python models/zoo/train_pseudo_label.py \
    --teacher_model_type "robodepth_dinov2_base" \
    --student_model_type "robodepth_vit_base" \
    --data_dir "/path/to/unlabeled/images" \
    --output_dir "./zoo_output"
```

**支持的模型**:
- `robodepth_resnet50` - ResNet-50
- `robodepth_resnet101` - ResNet-101
- `robodepth_dinov2_base` - DINOv2 ViT-Base
- `robodepth_dinov2_large` - DINOv2 ViT-Large
- `robodepth_vit_small` - Vision Transformer Small
- `robodepth_vit_base` - Vision Transformer Base

**核心功能**:
- 灵活的教师 - 学生模型组合
- BerHu、MSE、SILog 多种损失函数
- 混合精度训练支持
- 自动学习率调度

---

## 通用训练参数

所有训练脚本支持以下通用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 未标注图像目录 | 必需 |
| `--output_dir` | 输出目录 | `./output` |
| `--num_epochs` | 训练轮数 | 10 |
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

预期输出应显示所有模块的测试通过（✓）。

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

如需完整功能（如实际推理），请根据对应模型的官方文档安装依赖。

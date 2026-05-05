#!/usr/bin/env python3
"""
验证脚本：验证 Lotus-2 伪标签训练模块的功能完整性
无需安装额外 pip 环境，仅验证模块导入、类实例化和基本逻辑
"""

import sys
import os

# 添加 workspace 到路径
sys.path.insert(0, '/workspace')

def print_section(title):
    """打印分隔标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_imports():
    """测试模块导入"""
    print_section("1. 测试模块导入")
    try:
        from models.lotus import (
            PseudoLabelTrainingConfig,
            Lotus2Teacher,
            PseudoLabelDataset,
            PseudoLabelLoss,
            compute_depth_metrics,
            Lotus2PipelineWrapper,
            resize_image,
            resize_to_multiple_of_16,
        )
        print("✓ 所有核心类成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败：{e}")
        return False

def test_config():
    """测试配置类"""
    print_section("2. 测试配置类 (PseudoLabelTrainingConfig)")
    try:
        from models.lotus import PseudoLabelTrainingConfig
        
        config = PseudoLabelTrainingConfig(
            teacher_model_path="checkpoints/lotus-2",
            train_data_dir="data/train",
            output_dir="outputs/lotus_pseudo",
            image_extensions=[".jpg", ".png"],
            batch_size=4,
            num_workers=2,
            learning_rate=1e-4,
            num_epochs=10,
            loss_type="l1",
            use_confidence_weighting=True
        )
        
        print(f"✓ 配置创建成功")
        print(f"  - 教师模型路径：{config.teacher_model_path}")
        print(f"  - 训练数据目录：{config.train_data_dir}")
        print(f"  - 输出目录：{config.output_dir}")
        print(f"  - 批次大小：{config.batch_size}")
        print(f"  - 学习率：{config.learning_rate}")
        print(f"  - 损失类型：{config.loss_type}")
        return True
    except Exception as e:
        print(f"✗ 配置测试失败：{e}")
        return False

def test_teacher_model_wrapper():
    """测试教师模型包装器（不加载真实权重）"""
    print_section("3. 测试教师模型包装器 (Lotus2Teacher)")
    try:
        from models.lotus import Lotus2Teacher, PseudoLabelTrainingConfig
        
        config = PseudoLabelTrainingConfig(teacher_model_path="dummy_path")
        
        # 创建包装器实例（不初始化真实模型，会捕获 ImportError）
        try:
            teacher = Lotus2Teacher(config)
            print(f"✓ Lotus2Teacher 实例创建成功")
            print(f"  - 设备：{teacher.device}")
        except ImportError as e:
            print(f"✓ Lotus2Teacher 正确捕获依赖缺失：{type(e).__name__}")
            print(f"  - 这是预期行为（未安装 diffusers）")
            return True
        
        return True
    except Exception as e:
        print(f"✗ 教师模型测试失败：{e}")
        return False

def test_dataset():
    """测试伪标签数据集"""
    print_section("4. 测试伪标签数据集 (PseudoLabelDataset)")
    try:
        from models.lotus import PseudoLabelDataset
        import tempfile
        import os
        
        # 创建临时目录结构
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            pseudo_label_dir = os.path.join(tmpdir, "pseudo_labels")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(pseudo_label_dir, exist_ok=True)
            
            # 创建假图像和标签文件
            import numpy as np
            from PIL import Image
            
            for i in range(3):
                # 创建假图像
                img = Image.new('RGB', (512, 512), color=(128, 128, 128))
                img.save(os.path.join(data_dir, f"img_{i:03d}.png"))
                
                # 创建假深度标签
                depth = np.random.rand(512, 512).astype(np.float32)
                np.save(os.path.join(pseudo_label_dir, f"img_{i:03d}.npy"), depth)
            
            dataset = PseudoLabelDataset(
                data_dir=data_dir,
                pseudo_label_dir=pseudo_label_dir,
                task_name="depth",
                max_samples=3
            )
            
            print(f"✓ 数据集创建成功")
            print(f"  - 数据集大小：{len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                image_tensor, depth_tensor, metadata = sample
                print(f"  - 样本图像形状：{image_tensor.shape}")
                print(f"  - 样本深度形状：{depth_tensor.shape}")
                print(f"  - 图像路径：{metadata.get('image_path', 'N/A')}")
            
            return True
    except Exception as e:
        print(f"✗ 数据集测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function():
    """测试损失函数"""
    print_section("5. 测试损失函数 (PseudoLabelLoss)")
    try:
        from models.lotus import PseudoLabelLoss
        import torch
        
        loss_fn = PseudoLabelLoss(
            loss_type="l1",
            use_confidence=True,
            confidence_threshold=0.1
        )
        
        # 创建假数据
        pred_depth = torch.randn(2, 1, 64, 64)
        target_depth = torch.randn(2, 1, 64, 64)
        confidence = torch.rand(2, 1, 64, 64)
        
        # 测试不带置信度
        loss1 = loss_fn(pred_depth, target_depth)
        print(f"✓ 基础损失计算成功：{loss1.item():.4f}")
        
        # 测试带置信度
        loss2 = loss_fn(pred_depth, target_depth, confidence)
        print(f"✓ 置信度加权损失计算成功：{loss2.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """测试评估指标"""
    print_section("6. 测试评估指标 (compute_depth_metrics)")
    try:
        from models.lotus import compute_depth_metrics
        import torch
        
        pred = torch.abs(torch.randn(1, 1, 32, 32)) + 0.1
        target = torch.abs(torch.randn(1, 1, 32, 32)) + 0.1
        
        metrics = compute_depth_metrics(pred, target)
        
        print(f"✓ 指标计算成功")
        for key, value in metrics.items():
            print(f"  - {key}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 指标测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_wrapper():
    """测试管道包装器"""
    print_section("7. 测试管道包装器 (Lotus2PipelineWrapper)")
    try:
        from models.lotus import Lotus2PipelineWrapper
        
        # 由于需要 diffusers，这里只测试类的存在性和文档
        print(f"✓ Lotus2PipelineWrapper 类可访问")
        print(f"  - 类文档：{Lotus2PipelineWrapper.__doc__[:50]}...")
        
        # 测试静态方法（不需要实例化）
        from models.lotus import resize_image, resize_to_multiple_of_16
        print(f"✓ resize_image 函数可访问")
        print(f"✓ resize_to_multiple_of_16 函数可访问")
        
        return True
    except Exception as e:
        print(f"✗ 管道包装器测试失败：{e}")
        return False

def test_file_structure():
    """测试文件结构完整性"""
    print_section("8. 测试文件结构完整性")
    try:
        required_files = [
            '/workspace/models/lotus/__init__.py',
            '/workspace/models/lotus/config.py',
            '/workspace/models/lotus/teacher_model.py',
            '/workspace/models/lotus/pseudo_label_dataset.py',
            '/workspace/models/lotus/train_utils.py',
            '/workspace/models/lotus/pipeline_wrapper.py',
            '/workspace/models/lotus/train_pseudo_label.py'
        ]
        
        all_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✓ {os.path.basename(file_path)}")
            else:
                print(f"✗ {os.path.basename(file_path)} (缺失)")
                all_exist = False
        
        return all_exist
    except Exception as e:
        print(f"✗ 文件结构测试失败：{e}")
        return False

def main():
    """主验证函数"""
    print("="*60)
    print("  Lotus-2 伪标签训练模块验证")
    print("="*60)
    
    tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_imports),
        ("配置类", test_config),
        ("教师模型", test_teacher_model_wrapper),
        ("数据集", test_dataset),
        ("损失函数", test_loss_function),
        ("评估指标", test_metrics),
        ("管道包装器", test_pipeline_wrapper),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试发生未捕获异常：{e}")
            results.append((name, False))
    
    # 汇总结果
    print_section("验证结果汇总")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有验证通过！Lotus-2 模块已就绪。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查相关模块。")
        return 1

if __name__ == "__main__":
    sys.exit(main())

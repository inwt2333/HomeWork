#!/usr/bin/env python3
"""
评估学生在验证集上的预测结果

使用方法:
    python evaluate_validation.py [pred_labels.npy]
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import sys

def evaluate_validation(pred_labels_file='pred_labels.npy'):
    """
    在验证集上评估预测结果
    """
    try:
        # 加载数据
        validation_indices = np.load('validation_indices.npy')
        validation_labels_true = np.load('validation_labels.npy')
        pred_labels_all = np.load(pred_labels_file)
        
        # 检查格式
        if pred_labels_all.shape != (300,):
            print(f"❌ 错误: pred_labels形状应该是(300,)，但得到{pred_labels_all.shape}")
            return None
        
        if pred_labels_all.min() < 0 or pred_labels_all.max() > 6:
            print(f"❌ 错误: 标签应该在[0, 6]范围内，但得到[{pred_labels_all.min()}, {pred_labels_all.max()}]")
            return None
        
        # 提取验证集的预测结果
        pred_labels_val = pred_labels_all[validation_indices]
        
        # 计算指标
        ari = adjusted_rand_score(validation_labels_true, pred_labels_val)
        nmi = normalized_mutual_info_score(validation_labels_true, pred_labels_val)
        
        # 输出结果
        print("=" * 60)
        print("                 验证集评估结果")
        print("=" * 60)
        print(f"验证集数据点数: {len(validation_indices)} / 300 ({len(validation_indices)/300*100:.0f}%)")
        print()
        print(f"  ARI (Adjusted Rand Index):        {ari:7.4f}")
        print(f"  NMI (Normalized Mutual Info):     {nmi:7.4f}")
        print()
        print("=" * 60)
        print("注意：这只是验证集得分，最终评分基于测试集（240个点）")
        print("=" * 60)
        
        return {'ari': ari, 'nmi': nmi}
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 文件未找到 - {e}")
        print("\n请确保以下文件存在:")
        print("  - validation_indices.npy (验证集索引)")
        print("  - validation_labels.npy (验证集标签)")
        print(f"  - {pred_labels_file} (你的预测结果)")
        return None
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return None


if __name__ == '__main__':
    # 命令行参数
    if len(sys.argv) > 1:
        pred_file = sys.argv[1]
    else:
        pred_file = 'pred_labels.npy'
    
    evaluate_validation(pred_file)


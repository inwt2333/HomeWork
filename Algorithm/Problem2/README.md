# AI1804 算法设计与分析 - P2练习课代码框架

## 📋 文件说明

本压缩包包含第2次课上练习的所有代码框架，共4个问题：

### 📁 文件列表

1. **`problem_2_1/`** - 问题2-1：基于数组的基础算法应用（子文件夹）
   - `problem_2_1_1_two_sum.py` - Two Sum问题
   - `problem_2_1_2_max_subarray.py` - 最大子数组和
   - `problem_2_1_3_peak_element.py` - 寻找峰值元素
   - `problem_2_1_4_majority_element.py` - 多数元素

2. **`problem_2_2_sorted_set.py`** - 问题2-2：集合操作实现
   - 基于有序数组的集合数据结构
   - 二分查找应用

3. **`problem_2_3_merge_sort_inversions.py`** - 问题2-3：归并排序的应用
   - 归并排序实现
   - 逆序对统计

4. **`problem_2_4_hospital_system.py`** - 问题2-4：医院诊室分配系统
   - 负载均衡算法
   - K路归并
   - 综合应用

## 🚀 使用方法

### 1. 环境要求
- Python 3.6 或更高版本
- 无需额外依赖包

### 2. 运行方式
每个文件都可以独立运行：

```bash
# 运行问题2-1（进入子文件夹）
cd problem_2_1
python problem_2_1_1_two_sum.py
python problem_2_1_2_max_subarray.py
python problem_2_1_3_peak_element.py
python problem_2_1_4_majority_element.py

# 运行问题2-2  
python problem_2_2_sorted_set.py

# 运行问题2-3
python problem_2_3_merge_sort_inversions.py

# 运行问题2-4
python problem_2_4_hospital_system.py
```

### 3. 代码结构
每个文件包含：
- **函数/类定义** - 需要实现的核心算法
- **TODO标记** - 明确标出需要完成的部分
- **测试代码** - 验证实现正确性的测试用例
- **详细注释** - 算法提示和复杂度要求

## ✅ 完成要求

### 必须完成的部分
- 所有标记为 `# TODO:` 的代码段
- 确保测试用例能正确运行
- 满足时间复杂度要求

### 代码质量要求
- 添加适当的注释说明关键逻辑
- 变量命名要有意义
- 代码结构清晰易读

## 📊 算法复杂度要求

| 问题 | 算法 | 时间复杂度 | 空间复杂度 |
|------|------|------------|------------|
| 2-1.1 | Two Sum | O(n log n) | O(n) |
| 2-1.2 | 最大子数组和 | O(n log n) | O(log n) |
| 2-1.3 | 寻找峰值 | O(log n) | O(log n) |
| 2-1.4 | 多数元素 | O(n log n) | O(log n) |
| 2-2 | 集合操作 | O(log n)查找, O(n)插入删除 | O(n) |
| 2-3 | 归并排序+逆序对 | O(n log n) | O(n) |
| 2-4 | 医院系统 | O(n×m + n log k) | O(n) |

## 🎯 学习目标

通过完成这些练习，您将掌握：

1. **基础算法技巧**
   - 双指针技术
   - 二分查找及其变体
   - 分治算法设计

2. **数据结构应用**
   - 数组操作优化
   - 链表遍历处理
   - 有序结构维护

3. **算法分析能力**
   - 时间复杂度分析
   - 空间复杂度优化
   - 算法正确性验证

---

**祝您学习愉快！** 🎉

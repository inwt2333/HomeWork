"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-1.2：最大子数组和

学生姓名：韩岳成
学号：524531910029
"""

def find_max_subarray_sum(arr):
    """
    找到数组中连续子数组的最大和
    
    定义：连续子数组是指数组中相邻元素组成的子序列。
    例如：数组[1, -3, 2, 1, -1]中，[2, 1]是连续子数组，[1, 2, -1]不是连续子数组。
    
    算法要求：
    - 使用分治法
    - 时间复杂度：O(n log n)
    - 空间复杂度：O(log n)
    
    参数：
    - arr: 输入数组
    
    返回：
    - 最大子数组和（整数）
    """
    if not arr:
        return 0
    
    def max_crossing_sum(arr, left, mid, right):
        """
        计算跨越中点的最大子数组和
        
        参数：
        - arr: 数组
        - left: 左边界
        - mid: 中点
        - right: 右边界
        
        返回：
        - 跨越中点的最大子数组和
        """
        # 实现跨越中点的最大和计算
        left_max = float('-inf')
        s = 0
        for i in range(mid, left - 1, -1):
            s += arr[i]
            if s > left_max:
                left_max = s

        right_max = float('-inf')
        s = 0
        for j in range(mid + 1, right + 1):
            s += arr[j]
            if s > right_max:
                right_max = s

        return left_max + right_max

    def max_subarray_divide_conquer(arr, left, right):
        """
        分治法求最大子数组和
        
        参数：
        - arr: 数组
        - left: 左边界
        - right: 右边界
        
        返回：
        - 最大子数组和
        """
        # 实现分治递归
        if left == right:
            return arr[left]
        mid = int((left + right) / 2)
        return max(max_subarray_divide_conquer(arr, left, mid), max_subarray_divide_conquer(arr, mid + 1, right), max_crossing_sum(arr, left, mid, right))
    
    # 实现主函数逻辑
    left = 0
    right = len(arr) - 1
    return max_subarray_divide_conquer(arr,left,right)

def find_max_subarray_naive(arr):
    """
    朴素方法求最大子数组和（用于验证结果）
    时间复杂度: O(n²)
    """
    if not arr:
        return 0
    
    max_sum = float('-inf')
    n = len(arr)
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += arr[j]
            max_sum = max(max_sum, current_sum)
    
    return max_sum

def main():
    """测试函数"""
    print("=== 问题2-1.2：最大子数组和 ===")
    
    # 测试用例1：经典测试
    print("\n1. 经典测试：")
    arr1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"数组: {arr1}")
    
    # 朴素方法验证
    naive_result = find_max_subarray_naive(arr1)
    print(f"朴素方法结果: {naive_result}")
    
    # 分治方法
    divide_result = find_max_subarray_sum(arr1)
    print(f"分治方法结果: {divide_result}")
    
    if naive_result == divide_result:
        print("✓ 结果一致")
    else:
        print("✗ 结果不一致")
    
    print(f"期望结果: 6 (子数组 [4, -1, 2, 1])")
    
    # 测试用例2：全负数
    print("\n2. 全负数测试：")
    arr2 = [-3, -1, -4, -2]
    print(f"数组: {arr2}")
    
    naive_result2 = find_max_subarray_naive(arr2)
    divide_result2 = find_max_subarray_sum(arr2)
    
    print(f"朴素方法结果: {naive_result2}")
    print(f"分治方法结果: {divide_result2}")
    
    if naive_result2 == divide_result2:
        print("✓ 结果一致")
    else:
        print("✗ 结果不一致")
    
    print(f"期望结果: -1 (单个元素 [-1])")
    
    # 测试用例3：单个元素
    print("\n3. 单个元素测试：")
    arr3 = [5]
    print(f"数组: {arr3}")
    
    naive_result3 = find_max_subarray_naive(arr3)
    divide_result3 = find_max_subarray_sum(arr3)
    
    print(f"朴素方法结果: {naive_result3}")
    print(f"分治方法结果: {divide_result3}")
    
    if naive_result3 == divide_result3:
        print("✓ 结果一致")
    else:
        print("✗ 结果不一致")
    
    # 测试用例4：空数组
    print("\n4. 空数组测试：")
    arr4 = []
    print(f"数组: {arr4}")
    
    divide_result4 = find_max_subarray_sum(arr4)
    print(f"分治方法结果: {divide_result4}")
    print("期望结果: 0")
    
    print("\n=== 算法要求 ===")
    print("时间复杂度: O(n log n) - T(n) = 2T(n/2) + O(n)")
    print("空间复杂度: O(log n) - 递归调用栈深度")
    print("核心技巧: 分治思想，考虑左半、右半、跨越中点三种情况")

if __name__ == "__main__":
    main()

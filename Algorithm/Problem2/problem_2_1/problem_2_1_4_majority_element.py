"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-1.4：多数元素

学生姓名：韩岳成
学号：524531910029
"""

def find_majority_element(arr):
    """
    找到数组中出现次数大于n/2的元素
    
    多数元素定义：在数组中出现次数大于 ⌊n/2⌋ 的元素
    注意：题目保证多数元素一定存在
    
    算法要求：
    - 使用分治法
    - 时间复杂度：O(n log n)
    - 空间复杂度：O(log n)
    
    参数：
    - arr: 输入数组
    
    返回：
    - 多数元素的值，如果不存在返回None
    """
    
    def count_occurrences(arr, start, end, target):
        """
        统计元素在指定范围内的出现次数
        
        参数：
        - arr: 数组
        - start: 起始索引
        - end: 结束索引
        - target: 目标元素
        
        返回：
        - target在arr[start:end+1]中的出现次数
        """
        # 实现计数功能
        count = 0
        for i in range(start, end + 1):
            if arr[i] == target:
                count += 1
        return count
    
    def majority_helper(arr, left, right):
        """
        递归查找多数元素的辅助函数
        
        参数：
        - arr: 数组
        - left: 左边界
        - right: 右边界
        
        返回：
        - 在arr[left:right+1]范围内的多数元素候选
        """
        if left == right:
            return arr[left]
        mid = (left + right) // 2
        left_candidate = majority_helper(arr, left, mid)
        right_candidate = majority_helper(arr, mid + 1, right)

        if left_candidate == right_candidate:
            return left_candidate

        # 统计两候选在当前区间的出现次数，选超过半数者
        left_count = count_occurrences(arr, left, right, left_candidate) if left_candidate is not None else 0
        right_count = count_occurrences(arr, left, right, right_candidate) if right_candidate is not None else 0
        mid = (right - left + 1) // 2
        if left_count > mid:
            return left_candidate
        if right_count > mid:
            return right_candidate
        return None
       
    
    # TODO: 实现主函数逻辑
    if not arr:
        return None
    return majority_helper(arr, 0, len(arr) - 1)

def find_majority_naive(arr):
    """
    朴素方法查找多数元素（用于验证结果）
    时间复杂度: O(n²)
    """
    if not arr:
        return None
    
    n = len(arr)
    for i in range(n):
        count = 0
        for j in range(n):
            if arr[j] == arr[i]:
                count += 1
        
        if count > n // 2:
            return arr[i]
    
    return None

def find_majority_voting(arr):
    """
    Boyer-Moore投票算法（用于对比）
    时间复杂度: O(n)，空间复杂度: O(1)
    """
    if not arr:
        return None
    
    # 第一阶段：找候选元素
    candidate = None
    count = 0
    
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    # 第二阶段：验证候选元素
    count = sum(1 for num in arr if num == candidate)
    return candidate if count > len(arr) // 2 else None

def main():
    """测试函数"""
    print("=== 问题2-1.4：多数元素 ===")
    
    # 测试用例1：基础测试
    print("\n1. 基础测试：")
    arr1 = [3, 2, 3, 3, 2, 3, 3]
    print(f"数组: {arr1}")
    print(f"数组长度: {len(arr1)}, n/2 = {len(arr1)//2}")
    
    naive_result = find_majority_naive(arr1)
    voting_result = find_majority_voting(arr1)
    divide_result = find_majority_element(arr1)
    
    print(f"朴素方法结果: {naive_result}")
    print(f"投票算法结果: {voting_result}")
    print(f"分治方法结果: {divide_result}")
    
    if naive_result == divide_result:
        print("✓ 分治方法结果正确")
    else:
        print("✗ 分治方法结果错误")
    
    # 统计验证
    if divide_result is not None:
        count = arr1.count(divide_result)
        print(f"元素 {divide_result} 出现次数: {count}")
        if count > len(arr1) // 2:
            print("✓ 确实是多数元素")
        else:
            print("✗ 不是多数元素")
    
    # 测试用例2：所有元素相同
    print("\n2. 所有元素相同测试：")
    arr2 = [1, 1, 1, 1, 1]
    print(f"数组: {arr2}")
    
    naive_result2 = find_majority_naive(arr2)
    divide_result2 = find_majority_element(arr2)
    
    print(f"朴素方法结果: {naive_result2}")
    print(f"分治方法结果: {divide_result2}")
    
    if naive_result2 == divide_result2:
        print("✓ 分治方法结果正确")
    else:
        print("✗ 分治方法结果错误")
    
    # 测试用例3：边界情况
    print("\n3. 边界情况测试：")
    arr3 = [2, 2, 1, 1, 1, 2, 2]
    print(f"数组: {arr3}")
    print(f"数组长度: {len(arr3)}, n/2 = {len(arr3)//2}")
    
    naive_result3 = find_majority_naive(arr3)
    divide_result3 = find_majority_element(arr3)
    
    print(f"朴素方法结果: {naive_result3}")
    print(f"分治方法结果: {divide_result3}")
    
    if naive_result3 == divide_result3:
        print("✓ 分治方法结果正确")
    else:
        print("✗ 分治方法结果错误")
    
    # 统计验证
    if divide_result3 is not None:
        count = arr3.count(divide_result3)
        print(f"元素 {divide_result3} 出现次数: {count}")
        if count > len(arr3) // 2:
            print("✓ 确实是多数元素")
        else:
            print("✗ 不是多数元素")
    
    # 测试用例4：单个元素
    print("\n4. 单个元素测试：")
    arr4 = [42]
    print(f"数组: {arr4}")
    
    divide_result4 = find_majority_element(arr4)
    print(f"分治方法结果: {divide_result4}")
    
    if divide_result4 == 42:
        print("✓ 单个元素正确识别为多数元素")
    else:
        print("✗ 单个元素应该是多数元素")
    
    # 测试用例5：无多数元素（理论上不会出现，因为题目保证存在）
    print("\n5. 空数组测试：")
    arr5 = []
    print(f"数组: {arr5}")
    
    divide_result5 = find_majority_element(arr5)
    print(f"分治方法结果: {divide_result5}")
    
    if divide_result5 is None:
        print("✓ 空数组正确返回None")
    else:
        print("✗ 空数组应该返回None")
    
    print("\n=== 算法要求 ===")
    print("时间复杂度: O(n log n) - T(n) = 2T(n/2) + O(n)")
    print("空间复杂度: O(log n) - 递归调用栈深度")
    print("核心技巧: 分治思想，候选元素验证")
    print("关键洞察: 多数元素必定是左半或右半的多数元素之一")
    
    print("\n=== 算法对比 ===")
    print("分治法: O(n log n) 时间, O(log n) 空间")
    print("投票法: O(n) 时间, O(1) 空间 (Boyer-Moore算法)")
    print("朴素法: O(n²) 时间, O(1) 空间")

if __name__ == "__main__":
    main()

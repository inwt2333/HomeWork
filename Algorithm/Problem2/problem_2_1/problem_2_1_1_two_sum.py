"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-1.1：Two Sum问题

学生姓名：韩岳成
学号：524531910029
"""

def two_sum_sorted(arr, target):
    """
    找到数组中和为目标值的两个数，返回它们的索引
    
    算法要求：
    - 使用排序+双指针的方法
    - 时间复杂度：O(n log n)
    - 空间复杂度：O(n)
    
    参数：
    - arr: 输入数组
    - target: 目标和
    
    返回：
    - 如果找到，返回两个数的原始索引 [index1, index2]
    - 如果没找到，返回 None
    """
    # TODO: 实现Two Sum算法
    sorted_arr = sorted(arr)
    left, right = 0, len(arr) - 1
    index1 = index2 = -1
    print(arr, sorted_arr)
    while left < right:
        current_sum = sorted_arr[left] + sorted_arr[right]
        if current_sum == target:
            for i in range(len(arr)):
                if index1 == -1 and arr[i] == sorted_arr[left]:
                    index1 = i
                if index2 == -1 and arr[i] == sorted_arr[right]:
                    index2 = i
                if (index1 != -1) and (index2 != -1):
                    return [index1, index2]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None
    

def main():
    """测试函数"""
    print("=== 问题2-1.1：Two Sum问题 ===")
    
    # 测试用例1：基础测试
    print("\n1. 基础测试：")
    arr1 = [2, 7, 11, 15]
    target1 = 9
    result1 = two_sum_sorted(arr1, target1)
    print(f"数组: {arr1}")
    print(f"目标: {target1}")
    print(f"结果: {result1}")
    
    # 验证结果
    if result1:
        actual_sum = arr1[result1[0]] + arr1[result1[1]]
        print(f"验证: {arr1[result1[0]]} + {arr1[result1[1]]} = {actual_sum}")
        if actual_sum == target1:
            print("✓ 测试通过")
        else:
            print("✗ 测试失败")
    else:
        print("✗ 未找到结果")
    
    # 测试用例2：无解情况
    print("\n2. 无解测试：")
    arr2 = [1, 2, 3, 4]
    target2 = 10
    result2 = two_sum_sorted(arr2, target2)
    print(f"数组: {arr2}")
    print(f"目标: {target2}")
    print(f"结果: {result2}")
    if result2 is None:
        print("✓ 正确识别无解情况")
    else:
        print("✗ 应该返回None")
    
    # 测试用例3：重复元素
    print("\n3. 重复元素测试：")
    arr3 = [3, 3, 4, 5]
    target3 = 6
    result3 = two_sum_sorted(arr3, target3)
    print(f"数组: {arr3}")
    print(f"目标: {target3}")
    print(f"结果: {result3}")
    
    if result3:
        actual_sum = arr3[result3[0]] + arr3[result3[1]]
        print(f"验证: {arr3[result3[0]]} + {arr3[result3[1]]} = {actual_sum}")
        if actual_sum == target3:
            print("✓ 测试通过")
        else:
            print("✗ 测试失败")
    
    print("\n=== 算法要求 ===")
    print("时间复杂度: O(n log n) - 排序占主导")
    print("空间复杂度: O(n) - 存储索引对")
    print("核心技巧: 排序 + 双指针")

if __name__ == "__main__":
    main()

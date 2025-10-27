"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-3：归并排序的应用

学生姓名：韩岳成
学号：524531910029
"""

def merge_sort_with_inversion_count(arr):
    """
    归并排序并统计逆序对数量
    参数: arr - 待排序的数组
    返回: (sorted_arr, inversion_count) - 排序后的数组和逆序对数量
    """
    
    def merge(left, right):
        """
        合并两个有序数组，并统计跨越逆序对
        参数: left, right - 两个有序数组
        返回: (merged_array, cross_inversions) - 合并后的数组和跨越逆序对数量
        """
        # 实现合并逻辑，同时统计逆序对
        merged = []
        l = r = 0
        cross_inversions = left[1] + right[1]
        # print(left, right)
        while(l < len(left[0]) and r < len(right[0])):
            if left[0][l] < right[0][r]:
                merged.append(left[0][l])
                l += 1
            else:
                merged.append(right[0][r])
                r += 1
                cross_inversions += len(left[0]) - l
        while(l < len(left[0])):
            merged.append(left[0][l])
            l += 1
        while(r < len(right[0])):
            merged.append(right[0][r])
            r += 1
        return (merged, cross_inversions)
    
    def merge_sort_helper(arr):
        """
        递归实现归并排序并统计逆序对
        参数: arr - 待排序的数组
        返回: (sorted_arr, inversion_count) - 排序后的数组和逆序对数量
        """
        # 实现归并排序的递归逻辑
        if len(arr) > 1:
            mid = int(len(arr) / 2)
            left = arr[:mid]
            right = arr[mid:]
            # print(left, right)
            return merge(merge_sort_helper(left), merge_sort_helper(right))
        else:
            return (arr, 0)
    
    return merge_sort_helper(arr)

def count_inversions_naive(arr):
    """
    朴素方法统计逆序对（用于验证结果）
    时间复杂度: O(n^2)
    """
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def main():
    """测试函数"""
    print("=== 问题2-3：归并排序的应用 ===")
    
    # 测试用例1
    print("\n1. 基础测试：")
    arr1 = [3, 1, 4, 2]
    print(f"原数组: {arr1}")
    
    # 朴素方法验证
    naive_count = count_inversions_naive(arr1.copy())
    print(f"朴素方法逆序对数量: {naive_count}")
    
    # 归并排序方法
    sorted_arr1, inv_count1 = merge_sort_with_inversion_count(arr1.copy())
    print(f"归并排序后: {sorted_arr1}")
    print(f"归并方法逆序对数量: {inv_count1}")
    print(f"结果一致: {naive_count == inv_count1}")
    
    # 测试用例2
    print("\n2. 复杂测试：")
    arr2 = [5, 4, 3, 2, 1]  # 完全逆序
    print(f"原数组: {arr2}")
    
    naive_count2 = count_inversions_naive(arr2.copy())
    sorted_arr2, inv_count2 = merge_sort_with_inversion_count(arr2.copy())
    
    print(f"归并排序后: {sorted_arr2}")
    print(f"朴素方法逆序对数量: {naive_count2}")
    print(f"归并方法逆序对数量: {inv_count2}")
    print(f"结果一致: {naive_count2 == inv_count2}")
    
    # 测试用例3
    print("\n3. 已排序测试：")
    arr3 = [1, 2, 3, 4, 5]  # 已排序
    print(f"原数组: {arr3}")
    
    naive_count3 = count_inversions_naive(arr3.copy())
    sorted_arr3, inv_count3 = merge_sort_with_inversion_count(arr3.copy())
    
    print(f"归并排序后: {sorted_arr3}")
    print(f"朴素方法逆序对数量: {naive_count3}")
    print(f"归并方法逆序对数量: {inv_count3}")
    print(f"结果一致: {naive_count3 == inv_count3}")
    
    # 性能对比（可选）
    print("\n4. 算法复杂度说明：")
    print("朴素方法时间复杂度: O(n²)")
    print("归并排序方法时间复杂度: O(n log n)")
    print("当数组规模较大时，归并排序方法显著更快")

if __name__ == "__main__":
    main()

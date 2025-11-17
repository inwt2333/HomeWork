"""
AI1804 算法设计与分析
第3次课上练习：哈希表与优先队列

问题3-2：K路归并与去重

请在TODO标记处填写代码
"""

import heapq

def merge_k_unsorted_lists_top_n(lists, n):
    """
    合并K个未排序列表，获取Top-N
    
    参数:
        lists: K个未排序列表，[(item_id, priority), ...]
        n: 返回前n个item
    
    返回:
        [(item_id, priority), ...] 前n个，按priority降序
    
    时间复杂度要求: O(N log K)
    
    提示:
        1. 先对每个列表heapify（分治）
        2. 使用全局优先队列合并K个堆
        3. 使用dict处理键冲突，保留最大priority
        4. 使用set保证item唯一性
    """
    if not lists or n <= 0:
        return []
    
    # Step 1: 对每个列表heapify
    heaps = []
    for lst in lists:
        heap = [(-priority, item_id) for item_id, priority in lst]  # 取负数实现最大堆
        heapq.heapify(heap)
        heaps.append(heap)
    
    # Step 2: 使用全局优先队列合并K个堆
    global_heap = []
    for i, heap in enumerate(heaps):
        if heap:
            priority, item_id = heapq.heappop(heap)
            heapq.heappush(global_heap, (priority, item_id, i))
    
    # Step 3: 使用dict处理键冲突，保留最大priority
    item_dict = {}
    result = []
    
    # Step 4: 使用set保证item唯一性
    seen = set()
    
    while global_heap and len(result) < n:
        priority, item_id, heap_index = heapq.heappop(global_heap)
        priority = -priority  # 还原正数优先级
        
        if item_id not in seen:
            seen.add(item_id)
            if item_id not in item_dict or priority > item_dict[item_id]:
                item_dict[item_id] = priority
                result.append((item_id, priority))
        
        if heaps[heap_index]:
            next_priority, next_item_id = heapq.heappop(heaps[heap_index])
            heapq.heappush(global_heap, (next_priority, next_item_id, heap_index))
    
    # 按priority降序排序结果
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题3-2：K路归并与去重 - 测试")
    print("=" * 60)
    
    # 测试1: K个未排序列表，获取Top-3
    print("\n=== 测试1: K个未排序列表，获取Top-3 ===")
    lists1 = [
        [("A", 100), ("C", 60), ("B", 80)],      # 未排序
        [("E", 50), ("A", 90), ("D", 85)],       # 未排序
        [("F", 40), ("B", 95), ("C", 70)]        # 未排序
    ]
    result1 = merge_k_unsorted_lists_top_n(lists1, 3)
    print(f"  输出: {result1}")
    print(f"  期望: [('A', 100), ('B', 95), ('D', 85)]")
    
    if (len(result1) == 3 and 
        result1[0] == ("A", 100) and 
        result1[1] == ("B", 95) and 
        result1[2] == ("D", 85)):
        print("  ✓ 测试1通过")
    else:
        print("  ✗ 测试1失败")
    
    # 测试2: 键冲突保留最大值
    print("\n=== 测试2: 键冲突保留最大值 ===")
    lists2 = [
        [("X", 50), ("Y", 30)],
        [("X", 100), ("Z", 20)],  # X出现，更大priority
        [("X", 30), ("W", 40)]     # X再次出现，更小priority
    ]
    result2 = merge_k_unsorted_lists_top_n(lists2, 2)
    print(f"  输出: {result2}")
    print(f"  期望: X的priority应该是100（最大）")
    
    if len(result2) >= 1 and result2[0] == ("X", 100):
        print("  ✓ 测试2通过")
    else:
        print("  ✗ 测试2失败")
    
    # 测试3: n大于总item数
    print("\n=== 测试3: n大于总item数 ===")
    lists3 = [
        [("A", 100)],
        [("B", 90)]
    ]
    result3 = merge_k_unsorted_lists_top_n(lists3, 10)
    print(f"  输出长度: {len(result3)}")
    print(f"  期望: 2（只有2个不同item）")
    
    if len(result3) == 2:
        print("  ✓ 测试3通过")
    else:
        print("  ✗ 测试3失败")
    
    # 测试4: 包含空列表
    print("\n=== 测试4: 包含空列表 ===")
    lists4 = [
        [("A", 100), ("B", 80)],
        [],  # 空列表
        [("C", 90), ("A", 70)]  # A重复，保留100
    ]
    result4 = merge_k_unsorted_lists_top_n(lists4, 3)
    print(f"  输出: {result4}")
    
    if len(result4) == 3 and ("A", 100) in result4:
        print("  ✓ 测试4通过")
    else:
        print("  ✗ 测试4失败")
    
    print("\n" + "=" * 60)


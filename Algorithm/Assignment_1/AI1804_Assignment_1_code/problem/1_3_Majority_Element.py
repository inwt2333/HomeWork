"""
Problem 1-3: Majority Element
EN: Return the element that appears more than floor(n/2) times (guaranteed to exist).
CN: 返回在数组中出现次数超过 ⌊n/2⌋ 的元素（保证存在）。
"""

class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        def count_occurrences(arr, start, end, target):
            count = 0
            for i in range(start, end + 1):
                if arr[i] == target:
                    count += 1
            return count
            
        def majority_helper(arr, left, right):
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
            
            
            
        if not nums:
            return None
        return majority_helper(nums, 0, len(nums) - 1)


            
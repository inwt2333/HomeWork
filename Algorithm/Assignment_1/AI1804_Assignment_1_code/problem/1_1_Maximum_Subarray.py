"""
Problem 1-1: Maximum Subarray
EN: Given an integer array, find the contiguous subarray with the largest sum and return that sum.
CN: 给定整数数组，找到和最大的连续子数组，并返回其和。
"""

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.find_max_subarray_sum(nums)

    def find_max_subarray_sum(self, arr):
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
        
        left = 0
        right = len(arr) - 1
        return max_subarray_divide_conquer(arr,left,right)
            

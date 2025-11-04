"""
Problem 1-4: Find the Duplicate Number
EN: Given n+1 integers in [1..n], find the single duplicate without modifying the array and using only O(1) extra space.
CN: 给定位于区间 [1..n] 的 n+1 个整数，在不修改数组且仅用 O(1) 额外空间的前提下找出唯一的重复数。
"""

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Floyd's Tortoise and Hare (Cycle Detection) Algorithm
        # Phase 1: Finding the intersection point of the two runners.
        tortoise = nums[0]
        hare = nums[0]
        
        while True:
            tortoise = nums[tortoise]  # Move tortoise by 1 step
            hare = nums[nums[hare]]    # Move hare by 2 steps
            if tortoise == hare:
                break
        
        # Phase 2: Finding the entrance to the cycle.
        tortoise = nums[0]
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        
        return hare
    

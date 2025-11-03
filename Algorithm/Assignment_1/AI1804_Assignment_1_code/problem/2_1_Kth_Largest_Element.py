"""
Problem 2-1: Kth Largest Element in an Array
EN: Return the k-th largest element in the array (not necessarily distinct) without fully sorting.
CN: 在不整体排序的情况下，返回数组中的第 k 大元素（不要求元素互异）。
"""

class Solution(object):
    def findKthLargest(self, nums, k):
        def quickselect(left, right, index):
            # 随机选pivot减少最坏情况概率
            pivot = nums[0]
            
            # 三路划分
            less, equal, greater = [], [], []
            for i in range(left, right + 1):
                if nums[i] > pivot:
                    greater.append(nums[i])
                elif nums[i] < pivot:
                    less.append(nums[i])
                else:
                    equal.append(nums[i])
            
            # 判断k在哪个区间
            if index < len(greater):
                return self.find_from_list(greater, index)
            elif index < len(greater) + len(equal):
                return pivot
            else:
                return self.find_from_list(less, index - len(greater) - len(equal))
        
        def find_from_list(lst, index):
            nums[:] = lst[:] 
            return quickselect(0, len(lst) - 1, index)
        
        self.find_from_list = find_from_list
        return quickselect(0, len(nums) - 1, k - 1)

        
        

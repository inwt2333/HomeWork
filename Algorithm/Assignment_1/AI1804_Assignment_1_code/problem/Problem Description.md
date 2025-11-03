# Algorithm Homework Problems
[中文（Chinese version）](./题目描述.md)
## 1-1: Maximum Subarray

Given an integer array `nums`, find the subarray with the largest sum, and return its sum.

### Examples
- Input: `nums = [-2,1,-3,4,-1,2,1,-5,4]`  → Output: `6`  → The subarray `[4,-1,2,1]` has the largest sum 6.
- Input: `nums = [1]` → Output: `1` → The subarray `[1]` has the largest sum 1.
- Input: `nums = [5,4,-1,7,8]` → Output: `23` → The subarray `[5,4,-1,7,8]` has the largest sum 23.

### Constraints
- `1 <= nums.length <= 1e5`
- `-1e4 <= nums[i] <= 1e4`

### Follow-up
If you have figured out the $O(n)$ solution (Kadane's algorithm), try coding another solution using divide and conquer.

---

## 1-2: Find Peak Element

A peak element is an element that is strictly greater than its neighbors. Given a 0-indexed integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks. Assume `nums[-1] = nums[n] = -∞`.

You must write an algorithm that runs in $O(\log n)$ time.

### Examples
- Input: `nums = [1,2,3,1]` → Output: `2` → `nums[2] = 3` is a peak.
- Input: `nums = [1,2,1,3,5,6,4]` → Output: `5` → `nums[5] = 6` is a peak (index `1` is also valid).

### Constraints
- $1 \le \text{nums.length} \le 1000$
- $-2^{31} \le \text{nums}[i] \le 2^{31}-1$
- $\text{nums}[i] \ne \text{nums}[i+1]$ for all valid $i$.

---

## 1-3: Majority Element

Given an array `nums` of size `n`, return the majority element — the element that appears more than `⌊n/2⌋` times. You may assume that the majority element always exists in the array.

### Examples
- Input: `nums = [3,2,3]` → Output: `3`
- Input: `nums = [2,2,1,1,1,2,2]` → Output: `2`

### Constraints
- $n = \text{nums.length}$
- $1 \le n \le 5\times 10^4$
- $-10^9 \le \text{nums}[i] \le 10^9$

### Follow-up
Could you solve the problem in linear time and in $O(1)$ space? (Boyer–Moore Voting)

---

## 1-4: Find the Duplicate Number

Given an integer array `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive, return the single repeated number. Solve without modifying the array and using only constant extra space.

### Examples
- Input: `nums = [1,3,4,2,2]` → Output: `2`
- Input: `nums = [3,1,3,4,2]` → Output: `3`
- Input: `nums = [3,3,3,3,3]` → Output: `3`

### Constraints
- $1 \le n \le 10^5$
- $\text{nums.length} = n + 1$
- $1 \le \text{nums}[i] \le n$
- All integers appear once except for exactly one integer which appears two or more times.

### Follow-up
- How can we prove at least one duplicate must exist? (Pigeonhole Principle)
- Can you solve it in linear runtime? (Floyd's Tortoise and Hare)

---

## 2-1: Kth Largest Element in an Array

Given an integer array `nums` and an integer `k`, return the `k`th largest element in the array (not necessarily distinct). Can you solve it without sorting?

### Examples
- Input: `nums = [3,2,1,5,6,4], k = 2` → Output: `5`
- Input: `nums = [3,2,3,1,2,4,5,5,6], k = 4` → Output: `4`

### Constraints
- $1 \le k \le \text{nums.length} \le 10^5$
- $-10^4 \le \text{nums}[i] \le 10^4$

---

## 3-1: Hospital Clinic Allocation System

Design a system to assign patients to `k` clinics and finally produce a single global queue for the pharmacy.

### Problem Background
-- There are `k` clinics and `n` registered patients labeled `1..n`.
-- Patients arrive in arbitrary order (arrival list), which is not sorted by label.
-- Assign patients to clinics so that:
  - Load is balanced: any two clinic sizes differ by at most `1`.
  - Within each clinic, patients are maintained in ascending order by label.
- Finally, merge all clinic queues into one globally sorted queue for the pharmacy.

### Tasks (Implement a HospitalSystem class)
- `assign_patients_to_clinics(arrivals: List[int], k: int) -> List[List[int]]`
  - Distribute patients while maintaining per-clinic sorted order and global load balance.
  - Tie-breaking when multiple clinics have the same current load: pick the clinic with the smallest index.
- `merge_clinic_queues(queues: List[List[int]]) -> List[int]`
  - Merge `k` sorted clinic queues into one sorted array using divide-and-conquer (pairwise merge).
  - Target complexity $O(n\log k)$.
- `process_hospital_queue(arrivals: List[int], k: int) -> List[int]`
  - Orchestrate the full process: assign → merge → return final queue.

### Contract
- Input: `arrivals` length `n`, elements are unique labels `1..n`; integer `k`.
- Output: A list of length `n` sorted ascending by patient label.
- Valid even when `k > n` (some clinics may be empty) and when `k = 1` (no distribution).

### Examples
1) `k = 2`, `arrivals = [4,1,3,2]`
   - Assignment (tie-break by smaller clinic index):
     - Clinic 0: `[1,3]`
     - Clinic 1: `[2,4]`
   - Merged: `[1,2,3,4]`

2) `k = 3`, `arrivals = [7,1,5,3,2,6,4]`
   - One valid balanced assignment (sizes 3,2,2):
     - Clinic 0: `[1,5,7]` → sorted → `[1,5,7]`
     - Clinic 1: `[2,6]`
     - Clinic 2: `[3,4]`
   - Merged: `[1,2,3,4,5,6,7]`

### Constraints
- $1 \le k \le 10^3$
- $1 \le n \le 10^5$
- Patients are uniquely labeled $1..n$.


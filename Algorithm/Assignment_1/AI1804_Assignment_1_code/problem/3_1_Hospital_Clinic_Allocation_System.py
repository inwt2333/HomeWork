"""
Problem 3-1: Hospital Clinic Allocation System
EN: Assign patients to k clinics with balanced loads and per-clinic ascending order; when loads tie, choose the smaller clinic index. Finally, merge all clinic queues into one globally sorted queue.
CN: 将病人分配到 k 个诊室，保持负载均衡且各诊室内部按编号升序；负载相同时优先小编号诊室；最后将各诊室队列合并为一个全局升序队列。
"""

class HospitalSystem(object):
	def assign_patients_to_clinics(self, arrivals, k):
		"""
		将到达序列中的病人分配到 k 个诊室，保证：
		1) 负载均衡：任意两个诊室人数之差不超过 1；
		2) 诊室内按病人编号升序；
		3) 负载相同时，分配到编号更小的诊室。

		:type arrivals: List[int]
		:type k: int
		:rtype: List[List[int]]
		"""

		if not arrivals or k == 0:
			return []
		def binary_search_insert(clinic, patient_id):
			# 使用二分查找在诊室内插入病人编号，保持升序
			left, right = 0, len(clinic)
			while left < right:
				mid = (left + right) // 2
				if clinic[mid] < patient_id:
					left = mid + 1
				else:
					right = mid
			clinic.insert(left, patient_id)

		if not arrivals or k == 0:
			return []
		self.clinics = [[] for _ in range(k)]  # k 个诊室
		for patient_id in arrivals:
			# 找到负载最小的诊室（负载相同时选编号更小的诊室）
			min_load_clinic_index = min(range(k), key=lambda i: len(self.clinics[i]))
			# 将病人按编号升序插入该诊室
			clinic = self.clinics[min_load_clinic_index]
			binary_search_insert(clinic, patient_id)
		return self.clinics
		

	def merge_clinic_queues(self, queues):
		"""
		使用分治（两两归并）将 k 个已排序的诊室队列合并为一个全局升序队列。
		目标时间复杂度：O(n log k)。

		:type queues: List[List[int]]
		:rtype: List[int]
		"""
		if not queues:
			return []
		
		def _merge_k_queues_divide_conquer(queues, start, end):
			# 分治法合并队列范围[start, end]
			if start == end:
				return queues[start]
			mid = (start + end) // 2
			left_merged = _merge_k_queues_divide_conquer(queues, start, mid)
			right_merged = _merge_k_queues_divide_conquer(queues, mid + 1, end)
			return _merge_two_queues(left_merged, right_merged)
		
		def _merge_two_queues(queue1, queue2):
			# 合并两个有序队列
			merged = []
			l = r = 0
			while(l < len(queue1) and r < len(queue2)):
				if queue1[l] < queue2[r]:
					merged.append(queue1[l])
					l += 1
				else:
					merged.append(queue2[r])
					r += 1
			while(l < len(queue1)):
				merged.append(queue1[l])
				l += 1
			while(r < len(queue2)):
				merged.append(queue2[r])
				r += 1
			return merged
		
		return _merge_k_queues_divide_conquer(queues, 0, len(queues) - 1)
		

	def process_hospital_queue(self, arrivals, k):
		"""
		主流程：分配 → 合并 → 返回最终全局队列。

		:type arrivals: List[int]
		:type k: int
		:rtype: List[int]
		"""
		clinics = self.assign_patients_to_clinics(arrivals, k)
		final_queue = self.merge_clinic_queues(clinics)
		return final_queue
		




        



    





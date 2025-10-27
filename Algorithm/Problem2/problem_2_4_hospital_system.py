"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-4：医院诊室分配系统

学生姓名：___________
学号：___________
"""

class Patient:
    """病人类"""
    def __init__(self, patient_id, name):
        self.patient_id = patient_id  # 挂号号码
        self.name = name
    
    def __str__(self):
        return f"病人{self.patient_id}({self.name})"
    
    def __repr__(self):
        return self.__str__()

class PatientNode:
    """病人链表节点"""
    def __init__(self, patient, next=None):
        self.patient = patient
        self.next = next

class HospitalSystem:
    """医院诊室分配系统"""
    def __init__(self, num_clinics):
        self.num_clinics = num_clinics
        self.clinics = [[] for _ in range(num_clinics)]  # k个诊室（数组）
        self.final_queue = []  # 最终取药队列
    
    def assign_patients_to_clinics(self, patient_list_head):
        """
        将病人分配到各个诊室，要求负载均衡且每个诊室内按号码有序
        
        算法要求：
        1. 负载均衡：选择当前病人数最少的诊室
        2. 有序维护：在选定诊室中按病人号码有序插入
        
        技术点：Linked List遍历 + Array操作 + 有序插入
        """
        # 实现病人分配算法
        current = patient_list_head
        while current:
            # 选择负载最小的诊室
            min_load_clinic_index = min(range(self.num_clinics), key=lambda i: len(self.clinics[i]))
            # 插入病人到选定诊室
            self._insert_patient_sorted(self.clinics[min_load_clinic_index], current.patient)
            current = current.next
        pass
    
    def _insert_patient_sorted(self, clinic, patient):
        """
        将病人按号码顺序插入到诊室中（辅助方法）
        参数: clinic - 诊室列表, patient - 病人对象
        """
        # 实现有序插入
        i = 0
        while(i < len(clinic)):
            if clinic[i].patient_id > patient.patient_id:
                break
            i += 1
        clinic.insert(i, patient)
        
        pass
    
    def merge_clinic_queues(self):
        """
        合并所有诊室的队列为最终取药队列，要求使用分治法
        
        算法要求：
        1. 使用分治思想实现k路归并
        2. 最终队列按病人号码从小到大排序
        
        技术点：Divide & Conquer + Recursion + Array合并
        """
        # TODO: 实现k路归并算法
        self.final_queue = self._merge_k_queues_divide_conquer(self.clinics, 0, self.num_clinics - 1)
        
        pass
    
    def _merge_k_queues_divide_conquer(self, queues, start, end):
        """
        分治法合并队列范围[start, end]（辅助方法）
        """
        if start == end:
            return queues[start]
        mid = (start + end) // 2
        left_merged = self._merge_k_queues_divide_conquer(queues, start, mid)
        right_merged = self._merge_k_queues_divide_conquer(queues, mid + 1, end)
        return self._merge_two_queues(left_merged, right_merged)
    
    def _merge_two_queues(self, queue1, queue2):
        """
        合并两个有序队列（辅助方法）
        """
        # TODO: 实现两队列合并
        merged = []
        l = r = 0
        # print(left, right)
        while(l < len(queue1) and r < len(queue2)):
            if queue1[l].patient_id < queue2[r].patient_id:
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
    
    def process_hospital_queue(self, patient_list_head):
        """
        主处理流程 - 综合运用所有技术（已实现，展示整体流程）
        """
        print("开始医院排队系统处理...")
        
        # 步骤1: 统计病人信息（Linked List遍历）
        print("步骤1: 统计到达病人")
        total_patients = self._count_patients(patient_list_head)
        print(f"   总病人数: {total_patients}")
        
        # 步骤2: 分配病人到诊室（Linked List + Array + Sorting）
        print("步骤2: 分配病人到诊室（负载均衡+有序维护）")
        self.assign_patients_to_clinics(patient_list_head)
        
        # 显示分配结果
        for i, clinic in enumerate(self.clinics):
            patient_ids = [p.patient_id for p in clinic]
            print(f"   诊室{i+1}: {len(clinic)}人, 号码: {patient_ids}")
        
        # 步骤3: 合并诊室队列（Divide & Conquer + Recursion）
        print("步骤3: 合并诊室队列（k路归并）")
        self.merge_clinic_queues()
        
        final_ids = [p.patient_id for p in self.final_queue]
        print(f"   最终取药队列: {final_ids}")
        
        print("处理完成！")
        return {
            'total_patients': total_patients,
            'clinic_loads': [len(clinic) for clinic in self.clinics],
            'final_queue_size': len(self.final_queue),
            'is_final_sorted': all(self.final_queue[i].patient_id <= 
                                 self.final_queue[i+1].patient_id 
                                 for i in range(len(self.final_queue)-1))
        }
    
    def _count_patients(self, head):
        """统计病人总数（已实现的辅助方法）"""
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        return count

def main():
    """测试函数"""
    print("=== 问题2-4：医院诊室分配系统 ===")
    
    # 创建病人到达序列（号码有序但到达顺序随机）
    patients_data = [(5, "王五"), (2, "李二"), (8, "赵八"), (1, "张一"), (7, "孙七")]
    
    print(f"病人到达顺序: {[f'{id}({name})' for id, name in patients_data]}")
    
    # 构建链表
    head = PatientNode(Patient(patients_data[0][0], patients_data[0][1]))
    current = head
    for patient_id, name in patients_data[1:]:
        current.next = PatientNode(Patient(patient_id, name))
        current = current.next
    
    # 创建3个诊室的医院系统
    hospital = HospitalSystem(3)
    
    # 执行处理流程
    report = hospital.process_hospital_queue(head)
    
    print(f"\n=== 处理结果分析 ===")
    print(f"总病人数: {report['total_patients']}")
    print(f"各诊室负载: {report['clinic_loads']}")
    print(f"最终队列大小: {report['final_queue_size']}")
    print(f"最终队列是否有序: {report['is_final_sorted']}")
    
    # 验证负载均衡
    loads = report['clinic_loads']
    max_load = max(loads)
    min_load = min(loads)
    print(f"负载均衡度: 最大负载{max_load}, 最小负载{min_load}, 差值{max_load - min_load}")
    
    if max_load - min_load <= 1:
        print("✓ 负载均衡良好")
    else:
        print("✗ 负载不均衡")
    
    if report['is_final_sorted']:
        print("✓ 最终队列排序正确")
    else:
        print("✗ 最终队列排序错误")

if __name__ == "__main__":
    main()

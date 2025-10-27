"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-2：集合操作实现

学生姓名：韩岳成
学号：524531910029
"""

class SortedArraySet:
    def __init__(self):
        """初始化集合"""
        self._data = []  # 存储数据的有序列表
    
    def _binary_search(self, x):
        """
        二分查找元素x的位置，返回(found, index)
        found: 是否找到元素
        index: 如果找到，返回元素位置；如果没找到，返回应该插入的位置
        """
        # TODO: 实现二分查找算法
        left, right = 0, len(self._data) - 1
        while left <= right:
            mid = int((left + right) / 2)
            if self._data[mid] == x:
                return (True, mid)
            elif self._data[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        return (False, left)
    
    def insert(self, x):
        """向集合中插入元素x"""
        found, index = self._binary_search(x)
        if found:
            return
        else:
            self._data.append(0)
            for i in range(len(self._data) - 2, index - 1, -1):
                self._data[i + 1] = self._data[i]
            self._data[index] = x
            return
    
    def delete(self, x):
        """从集合中删除元素x，返回True如果成功，False如果元素不存在"""
        # TODO: 实现删除操作
        found, index = self._binary_search(x)
        if not found:
            return False
        else:
            for i in range(index, len(self._data) - 1):
                self._data[i] = self._data[i + 1]
            del self._data[len(self._data) - 1]
            return True

    
    def find(self, x):
        """查找元素x是否在集合中"""
        # TODO: 实现查找操作
        return self._binary_search(x)[0]
    
    def size(self):
        """返回集合大小"""
        return len(self._data)
    
    def to_list(self):
        """返回集合的有序列表表示"""
        return self._data.copy()
    
    def is_empty(self):
        """判断集合是否为空"""
        return len(self._data) == 0
    
    def __str__(self):
        """字符串表示"""
        return f"SortedSet({self._data})"

def main():
    """测试函数"""
    print("=== 问题2-2：集合操作实现 ===")
    
    # 创建集合并测试
    s = SortedArraySet()
    
    # 测试插入操作（包含重复元素）
    print("\n1. 插入测试：")
    test_data = [5, 2, 8, 1, 9, 3, 1, 2]  # 注意有重复元素
    print(f"插入数据: {test_data}")
    
    for x in test_data:
        s.insert(x)
        print(f"插入 {x} 后，集合: {s.to_list()}")
    
    print(f"最终集合: {s.to_list()}")
    print(f"集合大小: {s.size()}")
    
    # 测试查找操作
    print("\n2. 查找测试：")
    search_items = [1, 4, 8, 10]
    for item in search_items:
        found = s.find(item)
        print(f"查找 {item}: {'找到' if found else '未找到'}")
    
    # 测试删除操作
    print("\n3. 删除测试：")
    delete_items = [2, 10, 5]
    for item in delete_items:
        success = s.delete(item)
        print(f"删除 {item}: {'成功' if success else '失败'}")
        print(f"删除后集合: {s.to_list()}")

if __name__ == "__main__":
    main()

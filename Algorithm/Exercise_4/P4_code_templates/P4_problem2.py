"""
AI1804 算法设计与分析 - 第4次课上练习
问题4-2：课程依赖关系与拓扑排序

请完成以下两个任务的实现
"""

from collections import deque

def dfs(graph , start):
    """
    深度优先搜索 (DFS) 算法

    参数:
    graph: 图的邻接表表示
    start: 起始顶点

    返回:
    ancestors : 字典，ancestors[v]是DFS树中顶点 v 的所有祖先
    """
    # 收集所有顶点（包含仅出现在依赖里的点）
    if graph == {}:
        return {}
    nodes = set(graph.keys())
    for u, vs in graph.items():
        nodes.update(vs)

    ancestors = {v: set() for v in nodes}
    visited = set()
    onstack = set()  # 递归栈用于环检测

    def visit(u):
        visited.add(u)
        onstack.add(u)
        for v in graph.get(u, []):
            # 更新 v 的祖先集合：u 以及 u 的所有祖先
            ancestors[u].add(v)
            ancestors[u].update(ancestors[v])

            if v not in visited:
                if not visit(v):
                    return False
            elif v in onstack:
                # 发现回边 => 有向环
                return False
        onstack.remove(u)
        return True

    # 先从 start 出发（若存在）
    if start in nodes:
        if not visit(start):
            return None
    # Full-DFS：补遍历其余未访问节点
    for v in nodes:
        if v not in visited:
            if not visit(v):
                return None
    return ancestors

# ========== 任务1：拓扑排序 ==========

def topological_sort(graph):
    """
    对有向图进行拓扑排序
    
    参数:
        graph: 有向图，邻接表表示 {vertex: [dependencies]}
               例如 {'A': ['B', 'C']} 表示A依赖于B和C（B和C是A的先修）
    
    返回:
        拓扑排序列表，如果存在环则返回None
    
    时间复杂度: O(|V| + |E|)
    """
    # 实现拓扑排序
    # 提示：
    # 1. 使用Full-DFS遍历所有顶点
    # 2. 在DFS过程中维护ancestors集合检测环
    # 3. 记录每个顶点的完成时间（finishing time）
    # 4. 按完成时间的逆序返回结果（后完成的先输出）
    if graph == {}:
        return []
    ancestors = dfs(graph, next(iter(graph)))

    if ancestors is None:
        return None  # 有环
    visited = set()
    finishing_order = []
    def dfs_visit(u):
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs_visit(v)
        finishing_order.append(u)
    for vertex in graph:
        if vertex not in visited:
            dfs_visit(vertex)

    return finishing_order


# ========== 任务2：课程学习计划 ==========

def course_plan(prerequisites, target_courses):
    """
    找出学习目标课程所需的所有课程及其拓扑排序
    
    参数:
        prerequisites: 课程依赖关系 {course: [prerequisite_courses]}
        target_courses: 目标课程列表
    
    返回:
        包含所有必需课程的拓扑排序列表，如果存在环则返回None
    
    时间复杂度: O(|V| + |E|)
    """
    # 实现课程学习计划
    # 提示：
    # 1. 从目标课程开始，使用DFS找出所有先修课程（反向图）
    # 2. 或者：构建只包含必需课程的子图，然后进行拓扑排序
    # 3. 确保返回的拓扑排序包含所有必需的课程
    
    ancestors = dfs(prerequisites, next(iter(prerequisites)) if prerequisites else None)
    if ancestors is None:
        return None  # 有环
    required_courses = set()
    for course in target_courses:
        required_courses.add(course)
        if course in ancestors:
            required_courses.update(ancestors[course])
    
    # 构建只包含必需课程的子图
    subgraph = {course: [prereq for prereq in prereqs if prereq in required_courses]
                for course, prereqs in prerequisites.items() if course in required_courses}
    
    return topological_sort(subgraph)


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题4-2：课程依赖关系与拓扑排序")
    print("=" * 60)
    
    # 测试1: 拓扑排序（无环）
    print("\n=== 测试1: 拓扑排序（无环） ===")
    prerequisites1 = {
        '数据结构': ['程序设计基础'],
        '算法设计': ['数据结构', '离散数学'],
        '操作系统': ['数据结构', '计算机组成原理'],
        '编译原理': ['数据结构', '算法设计'],
        '数据库': ['数据结构'],
        '程序设计基础': [],
        '离散数学': [],
        '计算机组成原理': []
    }
    
    result1 = topological_sort(prerequisites1)
    print(f"拓扑排序结果: {result1}")
    
    # 验证：检查每条边的顺序是否正确
    if result1:
        pos = {course: i for i, course in enumerate(result1)}
        for course, deps in prerequisites1.items():
            for dep in deps:
                assert pos[dep] < pos[course], \
                    f"错误：{dep}应该在{course}之前"
        print("✓ 拓扑排序验证通过")
    else:
        print("✗ 拓扑排序失败（检测到环）")
    
    # 测试2: 检测环
    print("\n=== 测试2: 检测环 ===")
    prerequisites2 = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A']  # 形成环：A -> B -> C -> A
    }
    
    result2 = topological_sort(prerequisites2)
    print(f"拓扑排序结果: {result2}")
    assert result2 is None, "应该检测到环"
    print("✓ 环检测测试通过")
    
    # 测试3: 课程学习计划
    print("\n=== 测试3: 课程学习计划 ===")
    target = ['编译原理', '操作系统']
    plan = course_plan(prerequisites1, target)
    print(f"学习计划: {plan}")
    
    # 验证：目标课程应该在计划中
    if plan:
        assert '编译原理' in plan
        assert '操作系统' in plan
        # 验证先修课程也在计划中
        assert '数据结构' in plan
        assert '程序设计基础' in plan
        assert '算法设计' in plan
        assert '计算机组成原理' in plan
        
        # 验证拓扑顺序
        pos = {course: i for i, course in enumerate(plan)}
        for course in plan:
            if course in prerequisites1:
                for dep in prerequisites1[course]:
                    if dep in plan:
                        assert pos[dep] < pos[course], \
                            f"错误：{dep}应该在{course}之前"
        
        print("✓ 课程学习计划验证通过")
    else:
        print("✗ 课程学习计划失败（检测到环）")
    
    # 测试4: 边界情况
    print("\n=== 测试4: 边界情况 ===")
    # 空图
    empty_graph = {}
    assert topological_sort(empty_graph) == []
    
    # 单顶点图
    single_graph = {'A': []}
    result_single = topological_sort(single_graph)
    assert result_single == ['A']
    
    # 无依赖的多个课程
    independent = {
        'A': [],
        'B': [],
        'C': []
    }
    result_indep = topological_sort(independent)
    assert len(result_indep) == 3
    assert set(result_indep) == {'A', 'B', 'C'}
    
    print("✓ 边界测试通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


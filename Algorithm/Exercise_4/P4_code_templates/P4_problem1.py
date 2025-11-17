"""
AI1804 算法设计与分析 - 第4次课上练习
问题4-1：社交网络中的最短路径与连通分量

请完成以下两个任务的实现
"""

from collections import deque

def bfs(graph , start):
    """
    广度优先搜索 (BFS) 算法

    参数:
        graph: 图的邻接表表示 {vertex: [neighbors]}
        start: 起始顶点

    返回:
    distances : 字典，distances[v]是从起点到顶点v的距离
    parents : 字典，parents[v]是最短路径中顶点v的父顶点
    """
    # Initialize
    distances = {}
    parents = {}
    visited = set()

    # Initialize source vertex
    distances[start] = 0
    parents[start] = None
    visited.add(start)

    # Use queue to store current level vertices
    queue = deque([start])

    # BFS main loop
    while queue:
        u = queue.popleft() # FIFO: first in , first out

        # Traverse all neighbors of u
        for v in graph.get(u, []):
            if v not in visited:
                # Discover new vertex
                visited.add(v)
                distances[v] = distances[u] + 1
                parents[v] = u
                queue.append(v)

    # Set distance to infinity for unvisited vertices
    for v in graph:
        if v not in distances :
            distances[v] = float('inf')

    return distances, parents

def reconstruct_path (parents , start , target):
    """
    从parent字典重构从start到target的路径

    参数:
        parents: parent字典
        start: 起始顶点
        target: 目标顶点

    返回:
        路径列表，如果不存在路径则返回None
    """
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parents.get(current)
    path.reverse()
    return path if path[0] == start else None

# ========== 任务1：最短路径查找 ==========

def shortest_path(graph, start, target):
    """
    查找从start到target的最短路径
    
    参数:
        graph: 无向图，邻接表表示 {vertex: [neighbors]}
        start: 起始顶点
        target: 目标顶点
    
    返回:
        (path, length) 元组，其中：
        - path: 最短路径的顶点列表，如果不存在路径则为None
        - length: 最短路径长度，如果不存在路径则为-1
    
    时间复杂度: O(|V| + |E|)
    """
    # 实现算法
    distances, parents = bfs(graph, start)
    path = reconstruct_path(parents, start, target)
    length = distances[target] if path is not None else -1
    return path, length

# ========== 任务2：连通分量分析 ==========

def connected_components(graph):
    """
    找出所有连通分量
    
    参数:
        graph: 无向图，邻接表表示 {vertex: [neighbors]}
    
    返回:
        连通分量列表，每个分量是一个顶点列表
    
    时间复杂度: O(|V| + |E|)
    """
    # 实现Full-BFS或Full-DFS
    # 提示：
    # 1. 维护一个visited集合记录所有已访问的顶点
    # 2. 对每个未访问的顶点，运行BFS或DFS
    # 3. 每次BFS/DFS访问的所有顶点构成一个连通分量

    visited = set()
    components = []
    for vertex in graph:
        if vertex not in visited:
            # Start a new component
            component = []
            queue = deque([vertex])
            visited.add(vertex)
            while queue:
                u = queue.popleft()
                component.append(u)
                for v in graph.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(component)
    return components



# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题4-1：社交网络中的最短路径与连通分量")
    print("=" * 60)
    
    # 测试1: 最短路径查找
    print("\n=== 测试1: 最短路径查找 ===")
    graph1 = {
        'Alice': ['Bob', 'Charlie'],
        'Bob': ['Alice', 'David'],
        'Charlie': ['Alice', 'Eve'],
        'David': ['Bob'],
        'Eve': ['Charlie'],
        'Frank': ['Grace'],
        'Grace': ['Frank'],
        'Henry': []
    }
    
    # 测试路径存在
    path1, length1 = shortest_path(graph1, 'Alice', 'Eve')
    print(f"Alice到Eve的最短路径: {path1}")
    print(f"路径长度: {length1}")
    assert path1 == ['Alice', 'Charlie', 'Eve'], f"路径不正确: {path1}"
    assert length1 == 2, f"期望长度2，实际{length1}"
    
    # 测试路径不存在
    path2, length2 = shortest_path(graph1, 'Alice', 'Frank')
    print(f"Alice到Frank的最短路径: {path2}")
    print(f"路径长度: {length2}")
    assert path2 is None, f"期望None，实际{path2}"
    assert length2 == -1, f"期望-1，实际{length2}"
    
    # 测试相同顶点
    path3, length3 = shortest_path(graph1, 'Alice', 'Alice')
    print(f"Alice到Alice的最短路径: {path3}")
    print(f"路径长度: {length3}")
    assert path3 == ['Alice'], f"期望['Alice']，实际{path3}"
    assert length3 == 0, f"期望0，实际{length3}"
    
    print("✓ 最短路径测试通过")
    
    # 测试2: 连通分量分析
    print("\n=== 测试2: 连通分量分析 ===")
    components = connected_components(graph1)
    print(f"连通分量数量: {len(components)}")
    print("连通分量:")
    for i, comp in enumerate(components, 1):
        print(f"  分量{i}: {sorted(comp)}")
    
    # 验证结果
    assert len(components) == 3, f"期望3个连通分量，实际{len(components)}"
    all_vertices = set()
    for comp in components:
        all_vertices.update(comp)
    assert all_vertices == set(graph1.keys()), "顶点集合不匹配"
    
    print("✓ 连通分量测试通过")
    
    # 测试3: 边界情况
    print("\n=== 测试3: 边界情况 ===")
    # 空图
    empty_graph = {}
    path_empty, length_empty = shortest_path(empty_graph, 'A', 'B')
    assert path_empty is None and length_empty == -1
    assert connected_components(empty_graph) == []
    
    # 单顶点图
    single_graph = {'A': []}
    path_single, length_single = shortest_path(single_graph, 'A', 'A')
    assert path_single == ['A'] and length_single == 0
    components_single = connected_components(single_graph)
    assert len(components_single) == 1 and components_single[0] == ['A']
    
    print("✓ 边界测试通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


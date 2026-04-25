def main():
    # 读取顶点数 n 和边数 m
    n, m = map(int, input().split())

    # 创建邻接表和反向邻接表
    adj = [[] for _ in range(n)]
    reverse_adj = [[] for _ in range(n)]

    # 读取边的信息，构建图和反向图
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u].append(v)
        reverse_adj[v].append(u)

    # 第一次 DFS：按完成时间顺序记录节点
    visited = [False] * n
    order = []

    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for u in range(n):
        if not visited[u]:
            dfs1(u)

    # 第二次 DFS：在反向图中找强连通分量
    visited2 = [False] * n
    components = []

    def dfs2(u, comp):
        visited2[u] = True
        comp.append(u)
        for v in reverse_adj[u]:
            if not visited2[v]:
                dfs2(v, comp)

    # 按照第一次 DFS 得到的顺序反向遍历
    order.reverse()
    for u in order:
        if not visited2[u]:
            comp = []
            dfs2(u, comp)
            components.append(comp)

    # 输出每个强连通分量
    for comp in components:
        print("{ " + " ".join(map(str, comp)) + " }")
    # 输出总个数
    print(len(components))


if __name__ == "__main__":
    main()

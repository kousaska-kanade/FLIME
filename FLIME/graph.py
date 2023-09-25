import networkx as nx
import matplotlib.pyplot as plt

# 创建图
G = nx.Graph()

# 增加节点
G.add_nodes_from([30,35,41])

# 增加权重，数据格式（节点1，节点2，权重）
e = [(30, 35, 0.11509), (30, 41, -0.00014), (35, 41, 0.12369)]
for k in e:
    G.add_edge(k[0], k[1], weight=k[2])

# 普通的画图方式
# nx.draw(G, with_labels=True)

# 生成节点位置序列
pos = nx.spring_layout(G,iterations=20)

# 重新获取权重序列
weights = nx.get_edge_attributes(G, "weight")

# 画节点图
nx.draw_networkx(G, pos, with_labels=True)
# 画权重图
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

# 展示
plt.show()
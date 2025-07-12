import dgl
import sys

if len(sys.argv) < 2:
    sys.exit()

file_path = sys.argv[1]

try:
    graph_list, _ = dgl.load_graphs(file_path)
    g = graph_list[0]

    print(f"--- 正在检查文件: {file_path} ---")
    print("\n图的基本信息:")
    print(g)

    print("\n节点数据 (ndata) 中包含的键:")
    for key in g.ndata.keys():
        print(f"- '{key}'")

    print("\n边数据 (edata) 中包含的键:")
    for key in g.edata.keys():
        print(f"- '{key}'")

    print("\n------")

except Exception as e:
    print(f"读取文件时发生错误: {e}")
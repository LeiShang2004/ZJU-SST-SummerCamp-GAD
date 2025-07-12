import dgl
import torch
import os
import sys
from tqdm import tqdm

def build_unigad_dataset(original_file_path, output_dir):
    print(f"--- 文件: {original_file_path} ---")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    base_name = os.path.basename(original_file_path)
    output_file_path = os.path.join(output_dir, f"{base_name}-els")

    if os.path.exists(output_file_path):
        print(f" {output_file_path} 已存在")
        return

    try:
        graph_list, _ = dgl.load_graphs(original_file_path)
        g = graph_list[0]
        print("原始图加载")

        if 'label' in g.ndata:
            # 如果原始键是 'label', 将其重命名为 'node_label'
            g.ndata['node_label'] = g.ndata.pop('label')
            print("发现并成功重命名 'label' -> 'node_label'")
        elif 'node_label' not in g.ndata:
            print("未找到标签键")
            return

        node_labels = g.ndata['node_label']
        print(f"节点标签 共 {len(node_labels)} 个")

        print("正在根据论文生成 'edge_label'...")
        u, v = g.edges()
        u_labels = node_labels[u]
        v_labels = node_labels[v]

        # 论文：如果两个端点都异常，则边为异常 (1 & 1 = 1)
        # 论文没有具体写 简化
        edge_labels = (u_labels & v_labels).long()

 
        g.edata['edge_label'] = edge_labels
        print(f"边标签生成完毕 共 {len(edge_labels)} 条边 其中 {torch.sum(edge_labels)} 条为异常")


        dgl.save_graphs(output_file_path, [g])
        print(f"数据集已保存到: {output_file_path}")

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    source_folder = '../datasets/source'
    target_folder = '../datasets/edge_labels'


    print(f"处理文件夹 '{source_folder}' 中的所有数据集...")

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            build_unigad_dataset(file_path, target_folder)
"""
@File    :   tfgad.py
@Author  :   leijian <leijian2004@outlook.com>
@Date    :   2025-07-11

TFGAD 算法
"""

import dgl
import torch
import os
import time
import csv
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import warnings
# 高效处理稀疏矩阵 SVD
import scipy.sparse as sp 
from scipy.sparse.linalg import svds

# 忽略版本警告
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_rec_at_k(scores, labels, k):
    """
    Rec@K
    """
    
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    
    # top K节点
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    #  K 个预测中 真异常
    tp_at_k = np.sum(labels[top_k_indices])
    
    # 总的异常点数量
    total_positives = np.sum(labels)
    
    if total_positives == 0: return 0.0
        
    # Rec@K
    return tp_at_k / total_positives


def tfgad_detect(X, A, k_a, k_s, eta):
    """
    TFGAD 节点的异常分数
    """
  
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(A, torch.Tensor): A = torch.tensor(A, dtype=torch.float32)
    
    # 属性矩阵
    # 随机化SVD
    if k_a >= min(X.shape): # 如果k_a过大，则退回使用标准SVD
        _, _, V_a_t = torch.linalg.svd(X, full_matrices=False)
        V_a = V_a_t[:k_a, :].T
    else:
        _, _, V_a = torch.svd_lowrank(X, q=k_a)
        
    # -邻接矩阵处理
    if k_s is None or not A.is_sparse or k_s <= 0:
        # 如果不进行结构分析 则结构部分的异常分数为0
        adj_proj_len = torch.zeros(X.shape[0], device=X.device)
    else:
        # SciPy 稀疏矩阵
        A = A.coalesce() # coalescing 确保稀疏表示唯一
        # PyTorch 稀疏张量 转 SciPy稀疏矩阵
        indices = A.indices().cpu().numpy()
        values = A.values().cpu().numpy()
        A_scipy_csc = sp.csc_matrix((values, indices), shape=A.shape)
        
        # svds 奇异值分解
        # k<矩阵的最小维度
        safe_k = min(k_s, min(A_scipy_csc.shape)-1)
        _, _, Vt_s = svds(A_scipy_csc, k=safe_k)
        
        V_s = torch.from_numpy(Vt_s).T.to(X.device)
        
        # 结构投影长度 稀疏矩阵 @ 密集矩阵
        adj_proj_len = (A @ V_s.float()).pow(2).sum(dim=1)
            
    # 论文 异常分数 = 属性重构误差 + 结构投影长度 / eta
    att_rec_err = (X - (X @ V_a) @ V_a.T).pow(2).sum(dim=1)
    anomaly_scores = att_rec_err + adj_proj_len / eta
    
    return anomaly_scores


if __name__ == '__main__':
    # 数据集的配置
    
    # paper
    configs = {
        # --- 注入异常数据集 ---
        'Cora':        {'k_a': 1,   'k_s': 5,   'eta': 10,    'feature_key': 'feat', 'reported_auc': 0.9867, 'reported_auprc': 0.8197},
        'Citeseer':    {'k_a': 1,   'k_s': 5,   'eta': 100,   'feature_key': 'feat', 'reported_auc': 0.9895, 'reported_auprc': 0.8364},
        'Pubmed':      {'k_a': 1,   'k_s': 35,  'eta': 100,   'feature_key': 'feat', 'reported_auc': 0.9828, 'reported_auprc': 0.5830},
        'ACM':         {'k_a': 1,   'k_s': 60,  'eta': 10,    'feature_key': 'feat', 'reported_auc': 0.9677, 'reported_auprc': 0.4337},
        'BlogCatalog': {'k_a': 1,   'k_s': 220, 'eta': 0.05,  'feature_key': 'feat', 'reported_auc': 0.8042, 'reported_auprc': 0.2750},
        
        # --- 真实异常数据集 ---
        # books 数据集的 k_s 为 None，表示不使用结构信息进行SVD
        'books':       {'k_a': 10,  'k_s': None,'eta': 200,   'feature_key': 'feat', 'reported_auc': 0.7010, 'reported_auprc': 0.0571},
        'reddit':      {'k_a': 10,  'k_s': 5,   'eta': 500,   'feature_key': 'feat', 'reported_auc': 0.6021, 'reported_auprc': 0.0423},
    }
    base_path = "../GADAM/data"

    results_summary = []
    all_files_in_dir = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

    for filename in sorted(all_files_in_dir):
        dataset_name = os.path.splitext(filename)[0]

        if dataset_name not in configs:
            print(f"\n====================== Skipping File: {filename} (no config found for '{dataset_name}') ======================")
            continue

        print(f"\n====================== Processing Dataset: {dataset_name.upper()} (from file: {filename}) ======================")
        graph_path = os.path.join(base_path, filename)
        params = configs[dataset_name]
        
        try:
            graphs, _ = dgl.load_graphs(graph_path)
            graph = graphs[0]
            
            feature_key = params['feature_key']
            X = graph.ndata[feature_key].float()
            
            # 构建邻接矩阵
            u, v = graph.edges()
            indices = torch.stack([u, v])
            values = torch.ones(graph.num_edges())
            shape = (graph.num_nodes(), graph.num_nodes())
            A_torch_sparse = torch.sparse_coo_tensor(indices, values, size=shape)

            # 获取超参数
            k_a, k_s, eta = params['k_a'], params['k_s'], params['eta']
            
            print(f"图加载成功: {graph.num_nodes()} 个节点, {graph.num_edges()} 条边。")
            print(f"使用超参数: k_a={k_a}, k_s={k_s}, eta={eta}")
            
            start_time = time.time()
            
            # TFGAD
            anomaly_scores = tfgad_detect(X, A_torch_sparse, k_a=k_a, k_s=k_s, eta=eta)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"耗时: {elapsed_time:.2f} 秒。")


            ground_truth_labels = graph.ndata['label']
            k_for_recall = int(ground_truth_labels.sum())
           
            auc_score = roc_auc_score(ground_truth_labels.cpu().numpy(), anomaly_scores.cpu().numpy())
            auprc_score = average_precision_score(ground_truth_labels.cpu().numpy(), anomaly_scores.cpu().numpy())
            rec_k_score = calculate_rec_at_k(anomaly_scores, ground_truth_labels, k=k_for_recall)
            
            results_summary.append({
                "Dataset": dataset_name,
                "AUROC": f"{auc_score:.4f}",
                "AUPRC": f"{auprc_score:.4f}",
                "Rec@K": f"{rec_k_score:.4f}",
                "K": k_for_recall,
                "Time(s)": f"{elapsed_time:.2f}",
                "k_a": k_a,
                "k_s": k_s,
                "eta": eta
            })
            
        except Exception as e:
            print(f"数据集'{filename}' 错误: {e}")
            results_summary.append({"Dataset": dataset_name, "AUROC": "Error", "AUPRC": "Error", "Rec@K": "Error"})

    if results_summary:
        csv_filename = "results_paper.csv"
        fieldnames = ["Dataset", "AUROC", "AUPRC", "Rec@K", "K", "Time(s)", "k_a", "k_s", "eta"]
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_summary)
        print(f"\n\n保存到文件: {csv_filename}")

        print("\n====================================== FINAL SUMMARY ======================================")
        header_str = f"{'Dataset':<15} | {'AUROC':<10} | {'AUPRC':<10} | {'Rec@K':<10} | {'Time(s)':<10} | {'Params (ka,ks,eta)'}"
        print(header_str)
        print("-" * len(header_str))
        for result in results_summary:
            rec_k_str = f"{result.get('Rec@K', 'N/A')} (K={result.get('K', 'N/A')})"
            params_str = f"({result.get('k_a')},{result.get('k_s')},{result.get('eta')})"
            print(f"{result.get('Dataset', 'N/A'):<15} | {result.get('AUROC', 'N/A'):<10} | {result.get('AUPRC', 'N/A'):<10} | {rec_k_str:<20} | {result.get('Time(s)', 'N/A'):<10} | {params_str}")
        print("==========================================================================================")
    else:
        print("?"*10)
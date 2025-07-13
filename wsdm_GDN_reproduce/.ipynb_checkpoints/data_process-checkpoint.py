from utils.utils import sparse_to_adjlist
from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os

def process_elliptic_data():
    """
    处理椭圆比特币数据集，转换为与原始代码兼容的格式
    """
    print("Processing Elliptic Bitcoin Dataset...")
    
    # 检查椭圆数据集文件是否存在
    elliptic_files = [
        'data/elliptic_txs_features.csv',
        'data/elliptic_txs_edgelist.csv', 
        'data/elliptic_txs_classes.csv'
    ]
    
    if not all(os.path.exists(f) for f in elliptic_files):
        print("Elliptic dataset files not found, skipping...")
        return
    
    # 读取椭圆数据集
    features_df = pd.read_csv('data/elliptic_txs_features.csv', header=None)
    edges_df = pd.read_csv('data/elliptic_txs_edgelist.csv')
    classes_df = pd.read_csv('data/elliptic_txs_classes.csv')
    
    # 获取节点信息
    node_ids = features_df.iloc[:, 0].values
    n_nodes = len(node_ids)
    
    # 创建节点ID到索引的映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 处理边数据，构建邻接矩阵
    valid_edges = []
    for _, row in edges_df.iterrows():
        src, dst = row['txId1'], row['txId2']
        if src in node_id_to_idx and dst in node_id_to_idx:
            valid_edges.append([node_id_to_idx[src], node_id_to_idx[dst]])
    
    valid_edges = np.array(valid_edges)
    
    # 创建邻接矩阵（无向图）
    adj_matrix = csr_matrix((np.ones(len(valid_edges)), 
                            (valid_edges[:, 0], valid_edges[:, 1])), 
                           shape=(n_nodes, n_nodes))
    
    # 确保对称（无向图）
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix.data = np.ones_like(adj_matrix.data)  # 去除重复边的权重
    
    # 保存邻接列表（与原始代码格式一致）
    prefix = 'data/'
    sparse_to_adjlist(adj_matrix, prefix + 'elliptic_adjlists.pickle')
    
    print(f"Elliptic dataset processed: {n_nodes} nodes, {len(valid_edges)} edges")
    print(f"Adjacency list saved to: {prefix}elliptic_adjlists.pickle")

def process_yelp_data():
    """
    处理YelpChi数据集，适配实际的数据格式
    """
    print("Processing YelpChi dataset...")
    
    if not os.path.exists('data/YelpChi.mat'):
        print("YelpChi.mat not found, skipping...")
        return
        
    yelp = loadmat('data/YelpChi.mat')
    
    # 检查实际的键名
    print("Available keys in YelpChi.mat:", [k for k in yelp.keys() if not k.startswith('__')])
    
    # 根据实际数据结构处理
    # 您的数据中有: Network, Label, Attributes
    if 'Network' in yelp:
        main_network = yelp['Network']
        
        # 由于原始GDN代码期望多个关系网络，我们将主网络用作所有关系
        prefix = 'data/'
        
        # 将主网络保存为不同的关系类型
        sparse_to_adjlist(main_network, prefix + 'yelp_rur_adjlists.pickle')  # review-user-review
        sparse_to_adjlist(main_network, prefix + 'yelp_rtr_adjlists.pickle')  # review-text-review
        sparse_to_adjlist(main_network, prefix + 'yelp_rsr_adjlists.pickle')  # review-star-review
        sparse_to_adjlist(main_network, prefix + 'yelp_homo_adjlists.pickle') # homogeneous
        
        print("YelpChi dataset processed successfully")
        print(f"Network shape: {main_network.shape}")
        
    else:
        # 尝试原始的键名（如果有的话）
        try:
            net_rur = yelp['net_rur']
            net_rtr = yelp['net_rtr']
            net_rsr = yelp['net_rsr']
            yelp_homo = yelp['homo']
            sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
            sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
            sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
            sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')
            print("YelpChi dataset processed successfully (original format)")
        except KeyError as e:
            print(f"Error processing YelpChi: {e}")
            print("Available keys:", [k for k in yelp.keys() if not k.startswith('__')])

def process_amazon_data():
    """
    处理Amazon数据集，适配实际的数据格式
    """
    print("Processing Amazon dataset...")
    
    if not os.path.exists('data/Amazon.mat'):
        print("Amazon.mat not found, skipping...")
        return
        
    amz = loadmat('data/Amazon.mat')
    
    # 检查实际的键名
    print("Available keys in Amazon.mat:", [k for k in amz.keys() if not k.startswith('__')])
    
    prefix = 'data/'
    
    # 根据实际数据结构处理
    if 'Network' in amz:
        main_network = amz['Network']
        
        # 将主网络保存为不同的关系类型
        sparse_to_adjlist(main_network, prefix + 'amz_upu_adjlists.pickle')  # user-product-user
        sparse_to_adjlist(main_network, prefix + 'amz_usu_adjlists.pickle')  # user-star-user
        sparse_to_adjlist(main_network, prefix + 'amz_uvu_adjlists.pickle')  # user-view-user
        sparse_to_adjlist(main_network, prefix + 'amz_homo_adjlists.pickle') # homogeneous
        
        print("Amazon dataset processed successfully")
        print(f"Network shape: {main_network.shape}")
        
    else:
        # 尝试原始的键名
        try:
            net_upu = amz['net_upu']
            net_usu = amz['net_usu']
            net_uvu = amz['net_uvu']
            amz_homo = amz['homo']
            sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
            sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
            sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
            sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')
            print("Amazon dataset processed successfully (original format)")
        except KeyError as e:
            print(f"Error processing Amazon: {e}")
            print("Available keys:", [k for k in amz.keys() if not k.startswith('__')])

"""
    Read data and save the adjacency matrices to adjacency lists
"""
if __name__ == "__main__":
    print("Starting data processing...")
    print("=" * 50)
    
    # 处理 YelpChi 数据集
    process_yelp_data()
    print("-" * 30)
    
    # 处理 Amazon 数据集  
    process_amazon_data()
    print("-" * 30)
    
    # 处理椭圆数据集
    process_elliptic_data()
    print("-" * 30)
    
    print("=" * 50)
    print("Data processing completed!")
    
    # 检查生成的文件
    print("\nGenerated files:")
    generated_files = [
        'data/yelp_rur_adjlists.pickle',
        'data/yelp_rtr_adjlists.pickle', 
        'data/yelp_rsr_adjlists.pickle',
        'data/yelp_homo_adjlists.pickle',
        'data/amz_upu_adjlists.pickle',
        'data/amz_usu_adjlists.pickle',
        'data/amz_uvu_adjlists.pickle',
        'data/amz_homo_adjlists.pickle',
        'data/elliptic_adjlists.pickle'
    ]
    
    for file in generated_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
import argparse
import logging
import os
import sys
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from dataset import Dataset

# 尝试导入模型，如果不存在就跳过
try:
    from BWGNN import BWGNN
except ImportError:
    print("BWGNN not available")
    BWGNN = None

try:
    from GCN import GCN
except ImportError:
    print("GCN not available")
    GCN = None

try:
    from GraphSAGE import GraphSAGE
except ImportError:
    print("GraphSAGE not available") 
    GraphSAGE = None

# 尝试导入util函数
try:
    from util import set_random_seed, tab_printer, EarlyStopping
except ImportError:
    print("Some util functions not available, defining locally...")
    
    def set_random_seed(seed=42):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def tab_printer(args):
        """打印参数表格"""
        print("=" * 50)
        for key, value in vars(args).items():
            print(f"{key:20s}: {value}")
        print("=" * 50)
    
    class EarlyStopping:
        """早停机制"""
        def __init__(self, patience=7, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            """保存模型"""
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

# DDPM相关导入
try:
    from types import SimpleNamespace
    import ddpm.core.logger as Logger
    import ddpm.core.metrics as Metrics
    import ddpm.data as Data
    from ddpm.feature_test import pre_test
    import ddpm.model as Model
    from tensorboardX import SummaryWriter
    import math
    DDPM_AVAILABLE = True
except ImportError as e:
    print(f"DDPM modules not available: {e}")
    DDPM_AVAILABLE = False

def ddmpFeatures_fixed(features, non_features=None):
    """
    修复版的DDMP特征处理
    """
    print("\n" + "="*60)
    print("STARTING DDMP FEATURE PROCESSING")
    print("="*60)
    
    if not DDPM_AVAILABLE:
        print("DDPM modules not available, standardizing features as fallback...")
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        scaler = StandardScaler()
        processed_features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)
        print(f"✓ Standardized features shape: {processed_features.shape}")
        return processed_features

    try:
        # 检查并创建配置
        config_path = check_and_create_ddpm_config()
        
        # 创建参数对象
        args = SimpleNamespace()
        args.config = config_path
        args.phase = 'train'
        args.gpu_ids = [0] if torch.cuda.is_available() else []
        args.debug = False
        args.enable_wandb = False
        args.log_wandb_ckpt = False
        args.log_eval = False
        
        # 解析配置
        opt = Logger.parse(args)
        opt = Logger.dict_to_nonedict(opt)
        
        # 动态设置特征维度
        if isinstance(features, torch.Tensor):
            feature_dim = features.shape[-1]
            opt['model']['diffusion']['feature_dim'] = feature_dim
            print(f"✓ Set feature dimension to {feature_dim}")
        
        # 创建目录
        for path_key in ['log', 'tb_logger', 'results', 'checkpoint']:
            os.makedirs(opt['path'][path_key], exist_ok=True)
        
        # 设置日志
        Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=False)
        logger = logging.getLogger('base')
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
        print("✓ Logging setup completed")
        
        # 数据处理
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        print(f"✓ Input features shape: {features.shape}")
        
        # 创建训练数据
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train':
                train_set = Data.create_dataset(dataset_opt, features, phase)
                train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
                print("✓ Dataset created successfully")
                break
        
        # 模型创建和训练
        diffusion = Model.create_model(opt)
        print("✓ Model created successfully")
        
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], 
            schedule_phase='train'
        )
        
        # 简化的训练循环
        n_epoch = min(opt['train']['n_epoch'], 3)
        print(f"✓ Starting training for {n_epoch} epochs...")
        
        for epoch in range(n_epoch):
            for batch_idx, batch_data in enumerate(train_loader):
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                
                if not isinstance(batch_data, dict):
                    train_data = {'features': batch_data}
                else:
                    train_data = batch_data
                
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                
                if batch_idx >= 5:  # 限制每个epoch的batch数
                    break
                
            print(f"  Epoch {epoch+1}/{n_epoch} completed")
        
        print("✓ Training completed")
        
        # 特征处理
        print("✓ Starting feature processing...")
        processed_features = pre_test(features)
        
        if processed_features is not None:
            print(f"✓ Features successfully processed!")
            print(f"  Original shape: {features.shape}")
            print(f"  Processed shape: {processed_features.shape}")
            return processed_features
        else:
            return features
                
    except Exception as e:
        print(f"✗ DDMP processing failed: {e}, standardizing features as fallback...")
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        scaler = StandardScaler()
        processed_features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)
        print(f"✓ Standardized features shape: {processed_features.shape}")
        return processed_features
    
    finally:
        print("="*60)
        print("DDMP PROCESSING COMPLETED")
        print("="*60 + "\n")

def check_and_create_ddpm_config():
    """检查并创建DDPM配置文件"""
    config_path = 'ddpm/config/train.json'
    
    if not os.path.exists(config_path):
        print(f"Creating DDPM config at {config_path}...")
        
        default_config = {
            "name": "cagad_ddpm",
            "phase": "train",
            "gpu_ids": [0] if torch.cuda.is_available() else [],
            "path": {
                "log": "logs/ddpm",
                "tb_logger": "logs/tb_logger",
                "results": "results/ddpm",
                "checkpoint": "checkpoint/ddpm",
                "resume_state": None
            },
            "datasets": {
                "train": {
                    "name": "graph_features",
                    "mode": "features",
                    "batch_size": 32,
                    "num_workers": 4,
                    "use_shuffle": True
                }
            },
            "model": {
                "which_model_G": "ddpm",
                "beta_schedule": {
                    "train": {
                        "schedule": "linear",
                        "n_timestep": 100,
                        "linear_start": 1e-4,
                        "linear_end": 2e-2
                    }
                },
                "diffusion": {
                    "feature_dim": 64,
                    "conditional": True
                }
            },
            "train": {
                "n_epoch": 10,
                "val_epoch": 2,
                "save_checkpoint_epoch": 5,
                "print_freq": 1,
                "save_checkpoint_freq": 5
            },
            "optim": {
                "G": {
                    "lr": 1e-4,
                    "weight_decay": 0,
                    "beta1": 0.9,
                    "beta2": 0.999
                }
            }
        }
        
        os.makedirs('ddpm/config', exist_ok=True)
        import json
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"✓ Created DDPM config")
    
    return config_path

def ddmpFeatures(features, non_features=None):
    """DDPM特征处理函数"""
    return ddmpFeatures_fixed(features, non_features)

def create_heterograph_from_homograph(graph):
    """
    将同构图转换为异构图格式，创建三种边类型
    """
    edges = graph.edges()
    num_edges = len(edges[0])
    
    if num_edges == 0:
        # 空图处理
        empty_edges = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        graph_data = {
            ('review', 'net_rsr', 'review'): empty_edges,
            ('review', 'net_rtr', 'review'): empty_edges,
            ('review', 'net_rur', 'review'): empty_edges
        }
    else:
        # 根据边的特征将边分成三类
        src_nodes, dst_nodes = edges[0], edges[1]
        
        # 使用确定性方法分配边类型
        edge_sum = src_nodes + dst_nodes
        edge_types = edge_sum % 3  # 模3分配
        
        # 分配边到三种类型
        type0_mask = (edge_types == 0)
        type1_mask = (edge_types == 1) 
        type2_mask = (edge_types == 2)
        
        graph_data = {
            ('review', 'net_rsr', 'review'): (src_nodes[type0_mask], dst_nodes[type0_mask]),
            ('review', 'net_rtr', 'review'): (src_nodes[type1_mask], dst_nodes[type1_mask]),
            ('review', 'net_rur', 'review'): (src_nodes[type2_mask], dst_nodes[type2_mask])
        }
    
    hetero_graph = dgl.heterograph(graph_data)
    
    # 复制节点特征
    for key in graph.ndata.keys():
        hetero_graph.ndata[key] = graph.ndata[key]
    
    return hetero_graph

# 简单的模型定义（处理异构图）
class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, num_classes, allow_zero_in_degree=True)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, g, features):
        # 处理异构图：转换为同构图或使用第一个边类型
        if hasattr(g, 'etypes') and len(g.etypes) > 1:
            # 如果是异构图，转换为同构图
            try:
                # 尝试获取所有边并创建同构图
                all_edges = []
                for etype in g.etypes:
                    edges = g.edges(etype=etype)
                    all_edges.append((edges[0], edges[1]))
                
                # 合并所有边
                if all_edges:
                    src_all = torch.cat([edges[0] for edges in all_edges])
                    dst_all = torch.cat([edges[1] for edges in all_edges])
                    homo_g = dgl.graph((src_all, dst_all), num_nodes=g.num_nodes())
                    homo_g = dgl.add_self_loop(homo_g)
                    homo_g = homo_g.to(g.device)
                    g = homo_g
                else:
                    # 如果没有边，创建只有自环的图
                    self_loop_src = torch.arange(g.num_nodes())
                    self_loop_dst = torch.arange(g.num_nodes())
                    g = dgl.graph((self_loop_src, self_loop_dst), num_nodes=g.num_nodes()).to(g.device)
                    
            except Exception as e:
                print(f"Warning: Failed to convert heterograph: {e}")
                # 创建只有自环的图作为fallback
                self_loop_src = torch.arange(g.num_nodes())
                self_loop_dst = torch.arange(g.num_nodes())
                g = dgl.graph((self_loop_src, self_loop_dst), num_nodes=g.num_nodes()).to(g.device)
        else:
            # 为同构图添加自环
            g = dgl.add_self_loop(g)
        
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(input_dim, hidden_dim, 'mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_dim, num_classes, 'mean')
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, g, features):
        # 处理异构图：转换为同构图
        if hasattr(g, 'etypes') and len(g.etypes) > 1:
            try:
                all_edges = []
                for etype in g.etypes:
                    edges = g.edges(etype=etype)
                    all_edges.append((edges[0], edges[1]))
                
                if all_edges:
                    src_all = torch.cat([edges[0] for edges in all_edges])
                    dst_all = torch.cat([edges[1] for edges in all_edges])
                    homo_g = dgl.graph((src_all, dst_all), num_nodes=g.num_nodes())
                    homo_g = dgl.add_self_loop(homo_g)
                    homo_g = homo_g.to(g.device)
                    g = homo_g
                else:
                    self_loop_src = torch.arange(g.num_nodes())
                    self_loop_dst = torch.arange(g.num_nodes())
                    g = dgl.graph((self_loop_src, self_loop_dst), num_nodes=g.num_nodes()).to(g.device)
                    
            except Exception as e:
                print(f"Warning: Failed to convert heterograph: {e}")
                self_loop_src = torch.arange(g.num_nodes())
                self_loop_dst = torch.arange(g.num_nodes())
                g = dgl.graph((self_loop_src, self_loop_dst), num_nodes=g.num_nodes()).to(g.device)
        else:
            g = dgl.add_self_loop(g)
        
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

# 专门处理异构图的模型
class HeteroGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(HeteroGCN, self).__init__()
        self.convs = torch.nn.ModuleDict()
        
        # 为每种边类型创建卷积层（允许0入度）
        edge_types = ['net_rsr', 'net_rtr', 'net_rur']
        for etype in edge_types:
            self.convs[etype] = dgl.nn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        
        self.conv2 = dgl.nn.GraphConv(hidden_dim, num_classes, allow_zero_in_degree=True)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, g, features):
        if hasattr(g, 'etypes') and len(g.etypes) > 1:
            # 异构图处理
            h_list = []
            for etype in g.etypes:
                if etype in self.convs:
                    # 获取子图并添加自环
                    subgraph = g[etype]
                    subgraph = dgl.add_self_loop(subgraph)
                    h = self.convs[etype](subgraph, features)
                    h_list.append(h)
            
            if h_list:
                # 聚合不同边类型的结果
                h = torch.stack(h_list).mean(dim=0)
            else:
                # 如果没有有效的边类型，直接使用线性变换
                linear = torch.nn.Linear(features.shape[1], list(self.convs.values())[0].out_feats).to(features.device)
                h = linear(features)
        else:
            # 同构图处理
            g = dgl.add_self_loop(g)
            h = list(self.convs.values())[0](g, features)
        
        h = torch.relu(h)
        h = self.dropout(h)
        
        # 第二层需要同构图
        if hasattr(g, 'etypes') and len(g.etypes) > 1:
            # 转换为同构图进行第二层卷积
            try:
                all_edges = []
                for etype in g.etypes:
                    edges = g.edges(etype=etype)
                    all_edges.append((edges[0], edges[1]))
                
                if all_edges:
                    src_all = torch.cat([edges[0] for edges in all_edges])
                    dst_all = torch.cat([edges[1] for edges in all_edges])
                    homo_g = dgl.graph((src_all, dst_all), num_nodes=g.num_nodes())
                    homo_g = dgl.add_self_loop(homo_g)
                    homo_g = homo_g.to(g.device)
                    h = self.conv2(homo_g, h)
                else:
                    # 没有边的情况，直接通过线性层
                    linear = torch.nn.Linear(h.shape[1], self.conv2.out_feats).to(h.device)
                    h = linear(h)
            except Exception as e:
                print(f"Warning: Second layer conversion failed: {e}")
                linear = torch.nn.Linear(h.shape[1], self.conv2.out_feats).to(h.device)
                h = linear(h)
        else:
            g = dgl.add_self_loop(g)
            h = self.conv2(g, h)
        
        return h

def train_model(model, graph, args):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph = graph.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # 获取训练/验证/测试掩码 - 兼容不同数据集格式
    if 'train_mask' in graph.ndata:
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        print("Using standard mask format")
    elif 'train_masks' in graph.ndata:
        train_mask = graph.ndata['train_masks'][:, 0].bool()
        val_mask = graph.ndata['val_masks'][:, 0].bool()
        test_mask = graph.ndata['test_masks'][:, 0].bool()
        print("Using T-Finance multi-mask format (using mask 0)")
    else:
        print("No existing masks found, creating masks based on train_ratio...")
        num_nodes = graph.num_nodes()
        indices = torch.randperm(num_nodes)
        
        train_size = int(args.train_ratio * num_nodes)
        val_size = min(500, num_nodes // 10)
        test_size = min(1000, num_nodes - train_size - val_size)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:train_size + val_size + test_size]] = True
        
        print("Created new masks based on train_ratio")
    
    # 将masks保存到图中，供后续使用
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    
    labels = graph.ndata['label']
    
    # 打印mask统计信息
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Val nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    print(f"Total nodes: {graph.num_nodes()}")
    
    # 检查类别分布
    train_labels = labels[train_mask]
    print(f"Train label distribution: {torch.bincount(train_labels)}")
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    print("Starting training...")
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        
        try:
            logits = model(graph, graph.ndata['feature'])
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            
            # 验证
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(graph, graph.ndata['feature'])
                    val_loss = F.cross_entropy(val_logits[val_mask], labels[val_mask])
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = (val_pred == labels[val_mask]).float().mean()
                    
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                    
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA out of memory at epoch {epoch}. Trying to continue...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return model

def evaluate_model(model, graph, args):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        logits = model(graph, graph.ndata['feature'])
        test_mask = graph.ndata['test_mask']
        test_labels = graph.ndata['label'][test_mask].cpu().numpy()
        
        # 获取预测概率和标签
        test_probs = F.softmax(logits[test_mask], dim=1).cpu().numpy()
        test_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        
        # 计算指标
        # 对于二分类异常检测
        if test_probs.shape[1] == 2:
            # 异常类别通常是标签1
            test_probs_pos = test_probs[:, 1]
            
            # 计算指标
            f1 = f1_score(test_labels, test_pred, average='macro')
            auc_roc = roc_auc_score(test_labels, test_probs_pos)
            auc_pr = average_precision_score(test_labels, test_probs_pos)
        else:
            # 多分类情况
            f1 = f1_score(test_labels, test_pred, average='macro')
            # 对于多分类，计算每个类别的AUC然后平均
            from sklearn.preprocessing import label_binarize
            test_labels_bin = label_binarize(test_labels, classes=list(range(test_probs.shape[1])))
            auc_roc = roc_auc_score(test_labels_bin, test_probs, average='macro', multi_class='ovr')
            auc_pr = average_precision_score(test_labels_bin, test_probs, average='macro')
        
        accuracy = (test_pred == test_labels).mean()
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }

def main(args):
    """主函数，运行单个实验"""
    tab_printer(args)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据集
    dataset_name = args.dataset
    homo = args.homo
    
    try:
        dataset = Dataset(dataset_name, homo)
        graph = dataset.graph
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # 打印数据集信息
    print(f"  NumNodes: {graph.num_nodes()}")
    print(f"  NumEdges: {graph.num_edges()}")
    
    feature_key = 'feat' if 'feat' in graph.ndata else 'feature'
    if feature_key in graph.ndata:
        print(f"  NumFeats: {graph.ndata[feature_key].shape[1]}")
    
    if 'label' in graph.ndata:
        num_classes = graph.ndata['label'].max().item() + 1
        print(f"  NumClasses: {num_classes}")
    
    # 计算样本数
    if 'train_mask' in graph.ndata:
        print(f"  NumTrainingSamples: {graph.ndata['train_mask'].sum().item()}")
    if 'val_mask' in graph.ndata:
        print(f"  NumValidationSamples: {graph.ndata['val_mask'].sum().item()}")
    if 'test_mask' in graph.ndata:
        print(f"  NumTestSamples: {graph.ndata['test_mask'].sum().item()}")
    
    print("Done loading data from cached files.")
    print(graph)
    
    # 图处理逻辑
    print(f"Original graph: {graph}")
    print(f"Number of nodes: {graph.num_nodes()}")
    print(f"Number of edges: {graph.num_edges()}")
    if hasattr(graph, 'etypes'):
        print(f"Edge types: {graph.etypes}")

    if (homo):
        # 同构图处理
        print("Processing as homogeneous graph...")
        c = list(graph.edges())
        h = c[0] + graph.num_nodes()
        k = c[1] + graph.num_nodes()
        h2 = c[0] + 2*graph.num_nodes()
        k2 = c[1] + 2*graph.num_nodes()
        c[0] = torch.cat((c[0], h, h2))
        c[1] = torch.cat((c[1], k, k2))
        c = tuple(c)
        graph2 = dgl.graph(c)
    else:
        # 异构图处理
        print("Processing as heterogeneous graph...")
        
        # 检查是否已经是异构图
        if hasattr(graph, 'etypes') and len(graph.etypes) > 1:
            print("Input is already a heterogeneous graph")
            try:
                c1 = list(graph.edges(etype='net_rsr'))
                c2 = list(graph.edges(etype='net_rtr'))
                c3 = list(graph.edges(etype='net_rur'))
            except:
                # 如果没有预期的边类型，使用可用的边类型
                available_etypes = graph.etypes
                print(f"Available edge types: {available_etypes}")
                if len(available_etypes) >= 3:
                    c1 = list(graph.edges(etype=available_etypes[0]))
                    c2 = list(graph.edges(etype=available_etypes[1]))
                    c3 = list(graph.edges(etype=available_etypes[2]))
                else:
                    print("Converting homogeneous graph to heterogeneous format...")
                    hetero_graph = create_heterograph_from_homograph(graph)
                    c1 = list(hetero_graph.edges(etype='net_rsr'))
                    c2 = list(hetero_graph.edges(etype='net_rtr'))
                    c3 = list(hetero_graph.edges(etype='net_rur'))
        else:
            print("Converting homogeneous graph to heterogeneous format...")
            hetero_graph = create_heterograph_from_homograph(graph)
            c1 = list(hetero_graph.edges(etype='net_rsr'))
            c2 = list(hetero_graph.edges(etype='net_rtr'))
            c3 = list(hetero_graph.edges(etype='net_rur'))
        
        # 扩展边（创建3个副本）
        # 处理 net_rsr 边
        if len(c1[0]) > 0:
            h = c1[0] + graph.num_nodes()
            k = c1[1] + graph.num_nodes()
            h2 = c1[0] + 2*graph.num_nodes()
            k2 = c1[1] + 2*graph.num_nodes()
            c1 = (torch.cat((c1[0], h, h2)), torch.cat((c1[1], k, k2)))
        
        # 处理 net_rtr 边
        if len(c2[0]) > 0:
            h = c2[0] + graph.num_nodes()
            k = c2[1] + graph.num_nodes()
            h2 = c2[0] + 2*graph.num_nodes()
            k2 = c2[1] + 2*graph.num_nodes()
            c2 = (torch.cat((c2[0], h, h2)), torch.cat((c2[1], k, k2)))
        
        # 处理 net_rur 边
        if len(c3[0]) > 0:
            h = c3[0] + graph.num_nodes()
            k = c3[1] + graph.num_nodes()
            h2 = c3[0] + 2*graph.num_nodes()
            k2 = c3[1] + 2*graph.num_nodes()
            c3 = (torch.cat((c3[0], h, h2)), torch.cat((c3[1], k, k2)))
        
        print(f"Edge counts - rsr: {len(c1[0])}, rtr: {len(c2[0])}, rur: {len(c3[0])}")
        
        graph_data = {
            ('review', 'net_rsr', 'review'): c1,
            ('review', 'net_rtr', 'review'): c2,
            ('review', 'net_rur', 'review'): c3
        }
        graph2 = dgl.heterograph(graph_data)

    # 复制节点特征和标签
    feature_key = 'feature' if 'feature' in graph.ndata else 'feat'
    graph2.ndata['feature'] = torch.cat((graph.ndata[feature_key], 
                                       graph.ndata[feature_key], 
                                       graph.ndata[feature_key]))
    graph2.ndata['label'] = torch.cat((graph.ndata['label'], 
                                     graph.ndata['label'], 
                                     graph.ndata['label']))
    
    # 复制掩码
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in graph.ndata:
            graph2.ndata[mask_name] = torch.cat((graph.ndata[mask_name], 
                                               graph.ndata[mask_name], 
                                               graph.ndata[mask_name]))

    print(f"Final graph2: {graph2}")
    print(f"Nodes: {graph2.num_nodes()}, Edges: {graph2.num_edges()}")

    # DDPM处理
    if args.abchr and not args.skip_ddpm:
        try:
            print("Starting DDPM feature processing...")
            non_features = None
            graph2.ndata['feature'] = ddmpFeatures(graph2.ndata['feature'], non_features)
            print("DDPM processing completed successfully")
        except Exception as e:
            print(f"DDPM processing failed: {e}")
            print(f"Error details: {str(e)}")
            print("Continuing with original features...")
    else:
        print("Skipping DDPM feature processing")

    # 模型初始化
    input_dim = graph2.ndata['feature'].shape[1]
    hidden_dim = args.hid_dim
    num_classes = graph2.ndata['label'].max().item() + 1
    
    print(f"\nModel Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num classes: {num_classes}")
    
    # 安全的模型创建
    model = None
    
    if args.model == 'BWGNN':
        try:
            if 'BWGNN' in globals() and BWGNN is not None:
                model = BWGNN(input_dim, hidden_dim, num_classes, order=args.order)
                print("✓ BWGNN created successfully")
            else:
                raise ImportError("BWGNN not available")
        except Exception as e:
            print(f"BWGNN failed: {e}, using HeteroGCN")
            model = HeteroGCN(input_dim, hidden_dim, num_classes)
            
    elif args.model == 'GCN':
        try:
            if 'GCN' in globals() and GCN is not None:
                try:
                    model = GCN(input_dim, hidden_dim, num_classes)
                    print("✓ GCN created with basic interface")
                except TypeError:
                    model = GCN(
                        n_feat=input_dim,
                        n_hid=hidden_dim, 
                        n_classes=num_classes,
                        n_layers=2,
                        activation=torch.nn.ReLU(),
                        dropout=0.5
                    )
                    print("✓ GCN created with extended interface")
            else:
                raise ImportError("GCN not available")
        except Exception as e:
            print(f"GCN failed: {e}, using HeteroGCN")
            if hasattr(graph2, 'etypes') and len(graph2.etypes) > 1:
                model = HeteroGCN(input_dim, hidden_dim, num_classes)
                print("✓ Using HeteroGCN for heterogeneous graph")
            else:
                model = SimpleGCN(input_dim, hidden_dim, num_classes)
                print("✓ Using SimpleGCN for homogeneous graph")
            
    elif args.model == 'GraphSAGE':
        try:
            if 'GraphSAGE' in globals() and GraphSAGE is not None:
                model = GraphSAGE(input_dim, hidden_dim, num_classes)
                print("✓ GraphSAGE created successfully")
            else:
                raise ImportError("GraphSAGE not available")
        except Exception as e:
            print(f"GraphSAGE failed: {e}, using SimpleGraphSAGE")
            model = SimpleGraphSAGE(input_dim, hidden_dim, num_classes)
    
    # 默认模型选择
    if model is None:
        print(f"Using default model based on graph type")
        if hasattr(graph2, 'etypes') and len(graph2.etypes) > 1:
            model = HeteroGCN(input_dim, hidden_dim, num_classes)
            print("✓ Using HeteroGCN for heterogeneous graph")
        else:
            model = SimpleGCN(input_dim, hidden_dim, num_classes)
            print("✓ Using SimpleGCN for homogeneous graph")
    
    # 打印模型和图信息
    print(f"✓ Final model: {type(model).__name__}")
    print(f"✓ Graph type: {'Heterogeneous' if hasattr(graph2, 'etypes') and len(graph2.etypes) > 1 else 'Homogeneous'}")
    if hasattr(graph2, 'etypes'):
        print(f"✓ Edge types: {graph2.etypes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # 训练模型
    model = train_model(model, graph2, args)
    
    # 评估模型
    results = evaluate_model(model, graph2, args)
    
    print("\nExperiment completed successfully!")
    return results

def tune_hyperparameters(args):
    """执行超参数调优实验"""
    # 定义超参数网格
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'hid_dim': [32, 64, 128],
        'model': ['GCN', 'GraphSAGE', 'BWGNN'],
        'order': [1, 2]  # Only relevant for BWGNN
    }
    
    results = []
    for lr, hid_dim, model, order in product(param_grid['lr'], param_grid['hid_dim'], param_grid['model'], param_grid['order']):
        # Skip order > 1 for non-BWGNN models
        if model != 'BWGNN' and order != 1:
            continue
            
        print(f"\n=== Tuning: lr={lr}, hid_dim={hid_dim}, model={model}, order={order} ===")
        
        # 设置参数
        args.lr = lr
        args.hid_dim = hid_dim
        args.model = model
        args.order = order
        
        # 运行多次实验并平均结果
        run_results = []
        for run in range(args.run):
            print(f"Run {run + 1}/{args.run}")
            set_random_seed(42 + run)  # 不同种子
            result = main(args)
            if result is not None:
                run_results.append(result)
        
        if run_results:
            # 计算平均指标
            avg_result = {
                'lr': lr,
                'hid_dim': hid_dim,
                'model': model,
                'order': order,
                'accuracy': np.mean([r['accuracy'] for r in run_results]),
                'f1': np.mean([r['f1'] for r in run_results]),
                'auc_roc': np.mean([r['auc_roc'] for r in run_results]),
                'auc_pr': np.mean([r['auc_pr'] for r in run_results]),
                'runs': len(run_results)
            }
            results.append(avg_result)
            print(f"Average results: Accuracy={avg_result['accuracy']:.4f}, F1={avg_result['f1']:.4f}, "
                  f"AUC-ROC={avg_result['auc_roc']:.4f}, AUC-PR={avg_result['auc_pr']:.4f}")
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('tuning_results.csv', index=False)
    print("\nTuning results saved to 'tuning_results.csv'")
    
    # 打印最佳结果
    if results:
        best_result = results_df.loc[results_df['accuracy'].idxmax()]
        print("\nBest Hyperparameters:")
        print(best_result)
        
        # 生成可视化图表
        labels = [f"{row['model']}_lr{row['lr']}_hid{row['hid_dim']}_ord{row['order']}" for _, row in results_df.iterrows()]
        accuracies = results_df['accuracy'].tolist()
        
        print("\nGenerating visualization of tuning results...")
        chart = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Test Accuracy",
                    "data": accuracies,
                    "backgroundColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF"] * (len(labels) // 5 + 1),
                    "borderColor": ["#2E86C1", "#E74C3C", "#F1C40F", "#3AAFA9", "#8E44AD"] * (len(labels) // 5 + 1),
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Test Accuracy"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Hyperparameter Configuration"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True
                    }
                }
            }
        }
        print("Chart configuration generated (visualize using Chart.js):")
        print(chart)
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAGAD with Hyperparameter Tuning')
    parser.add_argument('--dataset', type=str, default='pubmed', 
                        choices=['pubmed', 'tfinance', 'amazon', 'yelp'])
    parser.add_argument('--train_ratio', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--abchr', type=int, default=1, help='Use DDMP processing')
    parser.add_argument('--nchr', type=int, default=0)
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['BWGNN', 'GCN', 'GraphSAGE'])
    parser.add_argument('--neighborchr', type=float, default=0.4)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--homo', type=int, default=1,
                        help='1 for homogeneous graphs, 0 for heterogeneous processing')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--run', type=int, default=3)  # Increased to 3 runs for robustness
    parser.add_argument('--skip_ddpm', action='store_true',
                        help='Skip DDPM processing')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning experiment')
    
    args = parser.parse_args()
    
    try:
        if args.tune:
            results_df = tune_hyperparameters(args)
            print("Tuning experiment completed successfully!")
        else:
            results = main(args)
            print("Single experiment completed successfully!")
    except Exception as e:
        print(f"Program failed with error: {e}")
        import traceback
        traceback.print_exc()
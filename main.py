import os
import torch
import torch.nn.functional as F
from Client import Client
from model import FDSE_GraphSAGE, FDSE_ResMLP
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


def split_client_data(data, val_ratio=0.1, test_ratio=0.1, device='cpu'):
    """
    数据预处理函数，将原始数据划分为训练、验证和测试集。
    """
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)

    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    train_data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    train_data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    train_data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    train_data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data


def load_all_clients(pyg_data_paths, encoder_params, decoder_params, training_params, device):
    """
    FDSE: 初始化客户端时使用 FDSE 版本的模型。
    """
    clients = []
    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, val_ratio=0.1, test_ratio=0.1, device=device)

        encoder = FDSE_GraphSAGE(
            input_dim=encoder_params['input_dim'],
            hidden_dim=encoder_params['hidden_dim'],
            output_dim=encoder_params['output_dim'],
            num_layers=encoder_params['num_layers'],
            dropout=encoder_params['dropout']
        )
        decoder = FDSE_ResMLP(
            input_dim=encoder_params['output_dim'] * 2,
            hidden_dim=decoder_params['hidden_dim'],
            num_layers=decoder_params['num_layers'],
            dropout=decoder_params['dropout']
        )

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            decoder=decoder,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)
    return clients


def average_state_dicts(state_dicts):
    """
    辅助函数：对模型参数进行加权平均。
    此函数用于聚合DFE参数。
    """
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key].cpu() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def flatten_state_dict(state_dict):
    """
    将state_dict展平为一维向量，用于计算相似性。
    """
    return torch.cat([v.view(-1) for v in state_dict.values()])


def similarity_aware_aggregation(ds_states, temperature=1.0):
    """
    FDSE: 完整实现论文中的相似性感知聚合。

    Args:
        ds_states (list): 包含所有客户端DSE参数字典的列表。
        temperature (float): 控制个性化程度的温度参数。

    Returns:
        list: 包含每个客户端个性化DSE参数字典的列表。
    """
    num_clients = len(ds_states)

    # 将每个客户端的DSE参数展平为向量
    flattened_params = [flatten_state_dict(sd) for sd in ds_states]
    V_l = torch.stack(flattened_params, dim=0)

    # 归一化向量以计算相似性
    normalized_V_l = F.normalize(V_l, p=2, dim=1)

    # 计算注意力分数 QK^T
    attention_scores = torch.matmul(normalized_V_l, normalized_V_l.T)

    # 应用softmax和温度参数
    attention_weights = F.softmax(attention_scores / temperature, dim=1)

    # 执行加权聚合，为每个客户端生成个性化参数
    aggregated_params = torch.matmul(attention_weights, V_l)

    # 将聚合后的向量重新转化为state_dict
    personalized_ds_states = []
    keys = list(ds_states[0].keys())

    for i in range(num_clients):
        current_state_dict = {}
        start_idx = 0
        for key in keys:
            param_shape = ds_states[0][key].shape
            param_size = ds_states[0][key].numel()

            # 从聚合后的向量中切分出对应的参数
            current_state_dict[key] = aggregated_params[i, start_idx:start_idx + param_size].view(param_shape).to(
                ds_states[i][key].device)
            start_idx += param_size
        personalized_ds_states.append(current_state_dict)

    return personalized_ds_states


def evaluate_all_clients(clients, use_test=False):
    """
    评估所有客户端模型的平均性能。
    """
    metrics = []
    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")
    avg_metrics = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average Metrics: Acc={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, "
          f"Prec={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
    return avg_metrics


if __name__ == "__main__":
    # 1. 配置路径与参数
    data_dir = "../Parsed_dataset/wd"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }
    decoder_params = {
        'hidden_dim': 128,
        'num_layers': 8,
        'dropout': 0.3
    }
    training_params = {
        'lr': 0.001,
        'weight_decay': 1e-4,
        'local_epochs': 5
    }

    num_rounds = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 初始化客户端和它们的模型
    clients = load_all_clients(pyg_data_files, encoder_params, decoder_params, training_params, device)
    num_clients = len(clients)

    best_f1 = -1

    # FDSE: 初始化全局DFE模型参数和个性化DSE模型参数列表
    global_df_state = clients[0].get_df_extractor_state()
    # 初始时，所有客户端的DSE参数都相同
    personalized_ds_states = [clients[0].get_ds_eraser_state() for _ in range(num_clients)]

    print("\n================ Federated Training Start ================\n")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 3. FDSE: 服务器分发**全局DFE模型**和**个性化DSE模型**给所有客户端
        df_states = []
        ds_states = []

        for i, client in enumerate(clients):
            # 将全局DFE和个性化DSE参数分发给客户端
            client.set_df_extractor_state(global_df_state)
            client.set_ds_eraser_state(personalized_ds_states[i])

            # 4. 每个客户端本地训练，并收集私有模型参数
            for _ in range(training_params['local_epochs']):
                loss = client.train()

            # 客户端返回训练后的DFE和DSE参数
            df_states.append(client.get_df_extractor_state())
            ds_states.append(client.get_ds_eraser_state())

        # 5. FDSE: 服务器分别聚合DFE和DSE参数
        # 5.1 对DFE参数进行公平平均聚合
        global_df_state = average_state_dicts(df_states)

        # 5.2 对DSE参数进行相似性感知聚合
        personalized_ds_states = similarity_aware_aggregation(ds_states, temperature=1.0)

        # 6. 联邦评估
        # 此时客户端模型包含服务器分发的最新全局DFE和个性化DSE
        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            # 此时的global_df_state和personalized_ds_states就是目前最好的模型
            best_df_state = global_df_state
            best_ds_states = personalized_ds_states
            print("===> New best global DFE and personalized DSE models saved.")

    print("\n================ Federated Training Finished ================\n")

    # 7. 最终模型评估
    print("\n================ Final Evaluation ================")
    for i, client in enumerate(clients):
        client.set_df_extractor_state(best_df_state)
        client.set_ds_eraser_state(best_ds_states[i])
    evaluate_all_clients(clients, use_test=True)
    
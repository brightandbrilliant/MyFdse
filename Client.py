import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

# 注意：需要从新的 model.py 文件中导入 FDSE 模型
from model import FDSE_GraphSAGE, FDSE_ResMLP


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cuda', lr=0.005, weight_decay=1e-4):
        """
        初始化 FDSE 客户端。
        客户端维护私有的FDSE编码器和解码器实例。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        # 优化器需要同时优化所有模块的参数，包括DFE和DSE
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # 存储从服务器接收的全局统计量
        self.global_stats = {}

    def train(self, lambda_reg=0.1):
        """
        执行客户端本地训练，并计算主任务损失和一致性正则化损失。

        Args:
            lambda_reg: 正则化项的权重系数，用于平衡两个损失。
        """
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        # 1. 生成负样本
        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        # 2. 模型前向传播，得到最终节点嵌入和中间层的输出
        z, intermediate_outputs = self.encoder(self.data.x, self.data.edge_index)

        # 3. 解码器预测边对分数
        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        # 4. 构造标签并计算主任务损失（BCE Loss）
        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=self.device),
            torch.zeros(neg_pred.size(0), device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
        task_loss = self.criterion(pred, labels)

        # 5. 计算一致性正则化损失 L_Con
        reg_loss = 0
        # 直接使用model.py返回的中间层输出
        for i, x_deskewed in enumerate(intermediate_outputs):

            # 计算本地DSE输出的均值和方差
            local_mean = x_deskewed.mean(dim=0)
            local_var = x_deskewed.var(dim=0)

            # 获取对应的全局DFE统计量
            # 这里的global_stats需要根据实际的键名来匹配，例如 'layer_0_stats'
            layer_name = f'layer_{i}_stats'
            if layer_name in self.global_stats:
                global_mean = self.global_stats[layer_name]['mean'].to(self.device)
                global_var = self.global_stats[layer_name]['var'].to(self.device)
            else:
                # 如果没有接收到全局统计量，使用默认值
                global_mean = torch.zeros_like(local_mean)
                global_var = torch.ones_like(local_var)

            # 论文公式 (6)
            reg_loss += (local_mean - global_mean).norm(p=2) ** 2
            reg_loss += (local_var.sqrt() - global_var.sqrt()).norm(p=2) ** 2

        total_loss = task_loss + lambda_reg * reg_loss

        # 6. 反向传播与优化
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def evaluate(self, use_test=False):
        """
        评估客户端模型性能。此函数与FedAvg版本基本相同。
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

            pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
            labels = torch.cat([
                torch.ones(pos_pred.size(0), device=self.device),
                torch.zeros(neg_pred.size(0), device=self.device)
            ])

            pred_label = (torch.sigmoid(pred) > 0.5).float()

            correct = (pred_label == labels).sum().item()
            acc = correct / labels.size(0)

            TP = ((pred_label == 1) & (labels == 1)).sum().item()
            FP = ((pred_label == 1) & (labels == 0)).sum().item()
            FN = ((pred_label == 0) & (labels == 1)).sum().item()

            recall = TP / (TP + FN + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return acc, recall, precision, f1

    def get_df_extractor_state(self):
        """
        获取DFE模块的参数状态，用于服务器端公平聚合。
        """
        df_states = {k: v for k, v in self.encoder.state_dict().items() if 'df_extractor' in k}
        df_states.update({k: v for k, v in self.decoder.state_dict().items() if 'df_extractor' in k})
        return df_states

    def get_ds_eraser_state(self):
        """
        获取DSE模块的参数状态，用于服务器端相似性感知聚合。
        """
        ds_states = {k: v for k, v in self.encoder.state_dict().items() if 'ds_eraser' in k}
        ds_states.update({k: v for k, v in self.decoder.state_dict().items() if 'ds_eraser' in k})
        return ds_states

    def set_df_extractor_state(self, state_dict, global_stats=None):
        """
        设置DFE模块的参数状态，并接收全局统计量。

        Args:
            state_dict: 服务器聚合后的DFE参数字典。
            global_stats: 服务器聚合后的DFE统计量字典。
        """
        # 分别为编码器和解码器准备一个完整的 state_dict
        encoder_state = self.encoder.state_dict()
        decoder_state = self.decoder.state_dict()

        # 用服务器提供的DFE参数更新各自的 state_dict
        for k, v in state_dict.items():
            if k in encoder_state:
                encoder_state[k] = v
            elif k in decoder_state:
                decoder_state[k] = v

        # 加载更新后的 state_dict
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)

        if global_stats:
            self.global_stats = global_stats

    def set_ds_eraser_state(self, state_dict):
        """
        设置DSE模块的参数状态，从服务器接收个性化模型。
        """
        # 分别为编码器和解码器准备一个完整的 state_dict
        encoder_state = self.encoder.state_dict()
        decoder_state = self.decoder.state_dict()

        # 用服务器提供的DSE参数更新各自的 state_dict
        for k, v in state_dict.items():
            if k in encoder_state:
                encoder_state[k] = v
            elif k in decoder_state:
                decoder_state[k] = v

        # 加载更新后的 state_dict
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)
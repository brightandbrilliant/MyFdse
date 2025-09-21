import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FDSE_Linear_Layer(nn.Module):
    """
    FDSE核心层，用于线性层。
    它将一个标准线性层分解为DFE和DSE。
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        初始化 FDSE 线性层。

        Args:
            in_dim: 输入特征的维度。
            out_dim: 输出特征的维度。
        """
        super(FDSE_Linear_Layer, self).__init__()

        # DFE (领域无关特征提取器): 使用一个线性层来充当此角色
        self.df_extractor = nn.Linear(in_dim, out_dim)

        # DSE (领域特定偏移擦除器): 使用BatchNorm1d来处理领域偏移
        self.ds_eraser = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        前向传播。

        Args:
            x: 输入特征张量。

        Returns:
            一个包含两个张量的元组：
            - 第一个张量是 DFE 的原始输出。
            - 第二个张量是经过 DSE 处理后的去偏移输出。
        """
        x_raw = self.df_extractor(x)
        x_deskewed = self.ds_eraser(x_raw)

        return x_raw, x_deskewed


class FDSE_GraphSAGE_Layer(nn.Module):
    """
    FDSE核心层，专门用于 GraphSAGE。
    它将一个 SAGEConv 层作为 DFE，并结合一个 BatchNorm 作为 DSE。
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        初始化 FDSE GraphSAGE 层。

        Args:
            in_dim: 输入特征的维度。
            out_dim: 输出特征的维度。
        """
        super(FDSE_GraphSAGE_Layer, self).__init__()

        # DFE (领域无关特征提取器): 使用SAGEConv来充当此角色
        self.df_extractor = SAGEConv(in_dim, out_dim)

        # DSE (领域特定偏移擦除器): 使用BatchNorm1d来处理领域偏移
        self.ds_eraser = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        前向传播。

        Args:
            x: 节点特征张量。
            edge_index: 边的索引张量。

        Returns:
            一个包含两个张量的元组：
            - 第一个张量是 DFE (SAGEConv) 的原始输出。
            - 第二个张量是经过 DSE 处理后的去偏移输出。
        """
        x_raw = self.df_extractor(x, edge_index)
        x_deskewed = self.ds_eraser(x_raw)

        return x_raw, x_deskewed


class FDSE_GraphSAGE(nn.Module):
    """
    FDSE 版本的 GraphSAGE 编码器。
    它使用 FDSE_GraphSAGE_Layer 来构建模型。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        初始化 FDSE 版本的 GraphSAGE 编码器。

        Args:
            input_dim: 输入特征的维度。
            hidden_dim: 隐藏层特征的维度。
            output_dim: 输出嵌入的维度。
            num_layers: 模型层数。
            dropout: dropout 比率。
        """
        super(FDSE_GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        # 首层和尾层保持线性或标准层，中间层使用FDSE_GraphSAGE_Layer
        self.initial_layer = FDSE_GraphSAGE_Layer(input_dim, hidden_dim)
        for i in range(num_layers - 2):
            self.layers.append(FDSE_GraphSAGE_Layer(hidden_dim, hidden_dim))
        self.final_layer = FDSE_GraphSAGE_Layer(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, x_: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        FDSE_GraphSAGE 的前向传播。

        Args:
            x_: 节点特征张量。
            edge_index: 边的索引张量。

        Returns:
            最终的节点嵌入张量。
        """
        # 首层前向传播
        x_raw, x_deskewed = self.initial_layer(x_, edge_index)
        x_ = x_deskewed
        x_ = F.relu(x_)
        x_ = F.dropout(x_, p=self.dropout, training=self.training)

        # 中间层迭代前向传播
        for layer in self.layers:
            x_raw, x_deskewed = layer(x_, edge_index)
            x_ = x_deskewed
            x_ = F.relu(x_)
            x_ = F.dropout(x_, p=self.dropout, training=self.training)

        # 最后一层前向传播
        x_raw, x_deskewed = self.final_layer(x_, edge_index)
        x_ = x_deskewed
        return x_


class FDSE_ResMLP(nn.Module):
    """
    FDSE 版本的 ResMLP 解码器。
    它使用 FDSE_Linear_Layer 来构建模型。
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        """
        初始化 FDSE 版本的 ResMLP 解码器。

        Args:
            input_dim: 输入特征的维度。
            hidden_dim: 隐藏层特征的维度。
            num_layers: 模型层数。
            dropout: dropout 比率。
        """
        super(FDSE_ResMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.initial_layer = FDSE_Linear_Layer(input_dim, hidden_dim)
        for i in range(num_layers):
            self.layers.append(FDSE_Linear_Layer(hidden_dim, hidden_dim))

        self.final_layer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        FDSE_ResMLP 的前向传播。

        Args:
            x_i: 第一个节点嵌入张量。
            x_j: 第二个节点嵌入张量。

        Returns:
            最终的边预测得分张量。
        """
        # 拼接两个节点嵌入作为输入
        x = torch.cat([x_i, x_j], dim=-1)

        # 初始层前向传播
        x_raw, x_deskewed = self.initial_layer(x)
        x = x_deskewed
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # FDSE 层迭代前向传播
        for layer in self.layers:
            x_raw, x_deskewed = layer(x)
            x = x_deskewed
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层前向传播，得到最终预测结果
        return self.final_layer(x)

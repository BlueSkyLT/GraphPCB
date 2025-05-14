from torch import nn
from .modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule)


MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        #print("Input to input_linear:", x.shape)
        x = self.input_linear(x)
        #print("After input_linear:", x.shape)
        x = self.dropout(x)
        x = self.act(x)

        for i, residual_module in enumerate(self.residual_modules):
            #print(f"Input to residual module {i}:", x.shape)
            x = residual_module(graph, x)
            #print(f"Output of residual module {i}:", x.shape)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)
        #print("Output of model:", x.shape)

        return x

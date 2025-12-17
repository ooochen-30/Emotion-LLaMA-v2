import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            Switch(),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out

class ConvModulePro(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvModulePro, self).__init__()

        self.conv_blocks = nn.ModuleList([])
        for i in range(4):
            self.conv_blocks.append(ConvBlock(input_channels, output_channels))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x) + x
        return x

class Switch(nn.Module):
    def __init__(self):
        super(Switch, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

    

class ConvProAttention(nn.Module):
    def __init__(self, dims, output_dim, layers='256,128', dropout=0.1):
        super(ConvProAttention, self).__init__()

        self.feats_prep = nn.ModuleList()
        self.internal_dim = int(layers.split(',')[-1]) 
        
        for i in range(len(dims)):
            self.feats_prep.append(self.MLP(dims[i], layers, dropout))

        self.features_count = len(dims)
        print(f"ConvProSeq: Input features count = {self.features_count}, Internal Dim = {self.internal_dim}")

        hiddendim = self.internal_dim * self.features_count
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)
        self.fc_att = nn.Linear(self.internal_dim, self.features_count)

        self.conv_entry = nn.Conv1d(hiddendim, self.internal_dim, kernel_size=1)
        self.conv_module = ConvModulePro(self.internal_dim, self.internal_dim)

        self.fc_final = nn.Linear(self.internal_dim, output_dim)

    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers_list = list(map(lambda x: int(x), layers.split(',')))

        for i in range(0, len(layers_list)):
            all_layers.append(nn.Linear(input_dim, layers_list[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers_list[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, feats):
        feats_hidden = []
        for i in range(self.features_count):
            feats_hidden.append(self.feats_prep[i](feats[i]))

        multi_hidden_flat = torch.cat(feats_hidden, dim=2)
        
        multi_hidden_stack = torch.stack(feats_hidden, dim=3)

        att_score = self.attention_mlp(multi_hidden_flat)
        att_score = self.fc_att(att_score) 
        att_score = torch.unsqueeze(att_score, 3) 

        fused_feat = torch.matmul(multi_hidden_stack, att_score).squeeze(3)

        conv_input = multi_hidden_flat.transpose(1, 2)
        
        conv_feat = self.conv_entry(conv_input)

        conv_feat = self.conv_module(conv_feat)

        conv_feat = conv_feat.transpose(1, 2)

        fused_feat = fused_feat + conv_feat

        out = self.fc_final(fused_feat)
        
        return out, None
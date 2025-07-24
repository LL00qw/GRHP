import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


class PositionalEncoding(nn.Module):
    def __init__(self, demb, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, demb, 2) * -(math.log(10000.0) / demb))
        pe = torch.zeros(max_len, 1, demb)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    '''
    Cross-modal attention module for combining visual and linguistic features
    '''

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.transformer_cross_modal = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )

    def forward(self, vis_feats, lang_feats):
        # Assuming vis_feats and lang_feats are of shape (S, N, E)
        combined_feats = torch.cat([vis_feats, lang_feats], dim=0)
        return self.transformer_cross_modal(combined_feats)


class SpatialPyramidPooling(nn.Module):
    """ 空间金字塔池化模块 """

    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        n, c, h, w = x.size()
        features = []
        for level in self.levels:
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = F.avg_pool2d(x, kernel_size, stride)
            features.append(F.interpolate(pooling, size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(features, dim=1)


class SelfAttention(nn.Module):
    """ 简单的自注意力层 """

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        value = self.value_conv(x).view(batch, -1, height * width)
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out + x


class CNNTransformerVisualEncoder(nn.Module):
    def __init__(self, dframe, rnn_hidden_size=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.dframe = dframe
        self.rnn_hidden_size = rnn_hidden_size
        self.spp = SpatialPyramidPooling(levels=[1, 2, 4])
        self.conv1 = nn.Conv2d(512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(64 * 7 * 7, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention = SelfAttention(512)
        self.transformer = nn.Transformer(
            d_model=self.dframe,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=self.rnn_hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.attention(x)
        x = self.spp(x)
        conv1_out = self.conv1(x)
        x = F.relu(self.bn1(conv1_out))
        conv2_out = self.conv2(x)
        x = F.relu(self.bn2(conv2_out))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1)
        x += self.create_positional_encoding(seq_len, self.dframe, x.device)
        output = self.transformer(x, x)
        return output, conv1_out, conv2_out

    def create_positional_encoding(self, seq_len, demb, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, demb, 2, device=device) * -(math.log(10000.0) / demb))
        pe = torch.zeros(seq_len, 1, demb, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe.squeeze(1)


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64, 7, 7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe
        self.d1 = nn.Linear(self.dhid, hshape[0] * hshape[1] * hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64 + 256, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32 + 64, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x, conv1_feat, conv2_feat):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        # 从编码器获取的特征
        conv1_feat = F.interpolate(conv1_feat, size=(14, 14), mode='nearest')
        conv2_feat = F.interpolate(conv2_feat, size=(56, 56), mode='nearest')

        x = self.upsample(x)
        x = torch.cat([x, conv1_feat], dim=1)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2_feat], dim=1)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))
        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')
        x = x.view(batch_size, seq_len, 1, self.pframe, self.pframe)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class GATFeatureExtractor(nn.Module):
    def __init__(self, feature_size):
        super(GATFeatureExtractor, self).__init__()
        self.gat1 = GATLayer(feature_size, feature_size)

    def forward(self, x, edge_index):
        return self.gat1(x, edge_index)


def create_linear_edge_index(num_nodes):
    edges = []
    for i in range(num_nodes):
        if i > 0:
            edges.append([i, i - 1])
        if i < num_nodes - 1:
            edges.append([i, i + 1])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)
        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = CNNTransformerVisualEncoder(dframe=dframe, rnn_hidden_size=128)
        self.cross_modal_attention = CrossModalAttention(d_model=128, nhead=8, dim_feedforward=dhid)
        d_model = 128
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=dhid)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        self.input_dropout = nn.Dropout(input_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(d_model, len(emb.weight))
        self.mask_dec = MaskDecoder(dhid=d_model, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.subgoal = nn.Linear(d_model, 1)
        self.progress = nn.Linear(d_model, 1)
        self.gat_feature_extractor = GATFeatureExtractor(1)
        nn.init.uniform_(self.go, -0.1, 0.1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, enc, frames, gold=None, max_decode=150):
        device = enc.device
        batch_size, seq_len, C, H, W = frames.size()

        # 编码视觉特征
        vis_feats, conv1_feat, conv2_feat = self.vis_encoder(frames)

        # 调整视觉特征维度
        if vis_feats.size(-1) != 128:
            fc = nn.Linear(vis_feats.size(-1), 128).to(device)
            vis_feats = fc(vis_feats.view(batch_size * seq_len, -1))
            vis_feats = vis_feats.view(seq_len, batch_size, -1)
        else:
            vis_feats = vis_feats.transpose(0, 1)

        # 准备Transformer的memory
        memory_lang = enc.transpose(0, 1)
        memory = self.cross_modal_attention(vis_feats, memory_lang)

        # 处理目标序列
        if gold is not None and self.teacher_forcing and self.training:
            tgt_seq = gold
        else:
            tgt_seq = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            tgt_seq[:, 0] = self.go.argmax().item()

        tgt_emb = self.emb(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        # Transformer解码
        outs = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        action_outs = self.actor(self.actor_dropout(outs))
        mask_outs = self.mask_dec(outs.transpose(0, 1), conv1_feat, conv2_feat)

        # 子目标和进度预测
        edge_index = create_linear_edge_index(seq_len).to(device)
        subgoal_outs = torch.sigmoid(self.subgoal(outs.transpose(0, 1)))
        progress_outs = torch.sigmoid(self.progress(outs.transpose(0, 1)))

        # GAT处理
        subgoal_outs_flat = subgoal_outs.view(-1, 1).to(device)
        progress_outs_flat = progress_outs.view(-1, 1).to(device)
        subgoal_outs_gat = self.gat_feature_extractor(subgoal_outs_flat, edge_index)
        subgoal_outs_gat = subgoal_outs_gat.view(batch_size, -1, 1)
        progress_outs_gat = self.gat_feature_extractor(progress_outs_flat, edge_index)
        progress_outs_gat = progress_outs_gat.view(batch_size, -1, 1)

        results = {
            'out_action_low': action_outs.transpose(0, 1),
            'out_action_low_mask': mask_outs,
            'out_subgoal': subgoal_outs_gat,
            'out_progress': progress_outs_gat,
        }
        return results

    def step(self, enc, frame, e_t, state_tm1=None):
        # 单步解码，用于推理
        device = enc.device
        batch_size = enc.size(0)

        # 编码当前帧
        frame = frame.unsqueeze(1)  # 添加序列维度
        vis_feats, conv1_feat, conv2_feat = self.vis_encoder(frame)
        seq_len = vis_feats.size(1)

        # 调整视觉特征维度
        if vis_feats.size(-1) != 128:
            fc = nn.Linear(vis_feats.size(-1), 128).to(device)
            vis_feats = fc(vis_feats.view(batch_size * seq_len, -1))
            vis_feats = vis_feats.view(seq_len, batch_size, -1)
        else:
            vis_feats = vis_feats.transpose(0, 1)

        # 准备memory
        memory_lang = enc.transpose(0, 1)
        memory = self.cross_modal_attention(vis_feats, memory_lang)

        # 目标序列（单步）
        tgt_emb = e_t.unsqueeze(0)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(1).to(device)

        # 解码
        outs = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        action_out = self.actor(self.actor_dropout(outs))
        action_out = action_out.squeeze(0)

        # 生成掩码（仅使用第一帧的卷积特征）
        conv1_feat = conv1_feat[:batch_size]
        conv2_feat = conv2_feat[:batch_size]
        mask_out = self.mask_dec(outs.transpose(0, 1), conv1_feat, conv2_feat)
        mask_out = mask_out.squeeze(1)

        # 子目标和进度预测
        subgoal_out = torch.sigmoid(self.subgoal(outs.transpose(0, 1)))
        progress_out = torch.sigmoid(self.progress(outs.transpose(0, 1)))

        return action_out, mask_out, (outs, memory), subgoal_out, progress_out




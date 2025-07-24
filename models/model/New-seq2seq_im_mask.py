import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data,Batch
import math

# 确保所有操作都在相同的设备上进行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 位置编码类的实现
class PositionalEncoding(nn.Module):
    def __init__(self, demb, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, demb, 2) * -(math.log(10000.0) / demb)).to(device)
        pe = torch.zeros(max_len, 1, demb).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NodeMatcher(nn.Module):
    """Module for matching nodes between scene and task graphs"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, task_nodes, scene_nodes):
        # task_nodes: [batch, task_nodes, dim]
        # scene_nodes: [batch, scene_nodes, dim]
        Q = self.query(task_nodes)
        K = self.key(scene_nodes)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        _, matched_idx = torch.max(attn_scores, dim=2)
        matched_scene_feats = self.batched_index_select(scene_nodes, 1, matched_idx)
        return matched_scene_feats, attn_scores

    def batched_index_select(self, input, dim, index):
        views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


class NodeGRUFusion(nn.Module):
    """Module for node-level GRU fusion"""

    def __init__(self, embed_dim):
        super().__init__()
        self.gru = nn.GRUCell(embed_dim, embed_dim)

    def forward(self, task_feats, matched_scene_feats):
        # task_feats: [batch, num_nodes, feat_dim]
        # matched_scene_feats: [batch, num_nodes, feat_dim]
        batch_size, num_nodes, feat_dim = task_feats.shape
        task_feats_flat = task_feats.reshape(-1, feat_dim)
        matched_scene_feats_flat = matched_scene_feats.reshape(-1, feat_dim)
        updated_flat = self.gru(task_feats_flat, matched_scene_feats_flat)
        updated_feats = updated_flat.view(batch_size, num_nodes, feat_dim)
        return updated_feats


class Module(Base):
    def __init__(self, args, vocab, scene_num_nodes, task_num_nodes):
        super().__init__(args, vocab)
        self.gamma = 2.0
        self.alpha = 0.25
        self.embed_dim = args.demb

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=args.demb, nhead=8, dim_feedforward=args.dhid)
        self.enc = TransformerEncoder(encoder_layer=encoder_layers, num_layers=6).to(device)
        self.pos_encoder = PositionalEncoding(args.demb, dropout=0.1).to(device)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # 新增：图编码器
        self.scene_encoder = Sequential('x, edge_index', [
            (GCNConv(scene_num_nodes, args.dhid), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(args.dhid, args.dhid), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(args.dhid, self.embed_dim), 'x, edge_index -> x')
        ]).to(device)

        self.task_encoder = Sequential('x, edge_index', [
            (GCNConv(task_num_nodes, args.dhid), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(args.dhid, args.dhid), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(args.dhid, self.embed_dim), 'x, edge_index -> x')
        ]).to(device)

        # 新增：图融合模块
        self.node_matcher = NodeMatcher(self.embed_dim)
        self.node_fusion = NodeGRUFusion(self.embed_dim)
        self.scene_fusion_layer = nn.Linear(self.embed_dim + self.embed_dim, self.embed_dim)
        self.task_fusion_layer = nn.Linear(self.embed_dim + self.embed_dim, self.embed_dim)
        self.scene_gru = nn.GRUCell(self.embed_dim, self.embed_dim)
        self.task_gru = nn.GRUCell(self.embed_dim, self.embed_dim)

        # 图特征与语言特征融合
        self.graph_lang_fusion = nn.Linear(self.embed_dim * 2 + self.embed_dim, self.embed_dim)

        # frame mask decoder
        decoder = vnn.ConvFrameMaskDecoderProgressMonitor
        self.dec = decoder(self.emb_action_low, args.dframe, 2 * args.dhid,
                           pframe=args.pframe,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # 新增：图状态
        self.scene_state = None
        self.task_state = None

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

    def featurize(self, batch, load_mask=True, load_frames=True):
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########
            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)
                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########
            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)

            # 新增：图数据
            feat['scene_graph'].append(ex.get('scene_graph', {}))
            feat['task_graph'].append(ex.get('task_graph', {}))

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))
                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)
                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########
            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])
                # low-level action mask
                if load_mask:
                    feat['action_low_mask'].append(
                        [self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])
                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

        # 处理图数据
        scene_graphs = [self.process_graph(g, device) for g in feat['scene_graph']]
        task_graphs = [self.process_graph(g, device) for g in feat['task_graph']]
        feat['scene_graph_batch'] = Batch.from_data_list(scene_graphs)
        feat['task_graph_batch'] = Batch.from_data_list(task_graphs)

        # 处理其他特征
        keys = list(feat.keys())
        for k in keys:
            if k in ['scene_graph', 'task_graph']:
                continue

            v = feat[k]
            if k == 'lang_goal_instr':
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                feat['lang_goal_instr_lengths'] = torch.tensor(seq_lengths, device=device)
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k == 'action_low_mask':
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if 'frames' in k else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        seq_lengths = [len(vv) for vv in feat['lang_goal_instr']]
        feat['seq_lengths'] = torch.tensor(seq_lengths, device=device)
        return feat

    def process_graph(self, graph_data, device):
        """处理图数据为PyTorch Geometric格式"""
        if not graph_data:
            # 处理空图情况
            return Data(x=torch.tensor([[1.0]], device=device),
                        edge_index=torch.tensor([[], []], dtype=torch.long, device=device))

        nodes = graph_data.get('nodes', [])
        links = graph_data.get('links', [])

        # 创建节点特征（单位矩阵）
        node_features = torch.eye(len(nodes), device=device)

        # 创建边索引
        node_indices = {node['id']: i for i, node in enumerate(nodes)}
        edge_index = []
        for link in links:
            if link['source'] in node_indices and link['target'] in node_indices:
                source = node_indices[link['source']]
                target = node_indices[link['target']]
                edge_index.append([source, target])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def serialize_lang_action(self, feat):
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]

    def decompress_mask(self, compressed_mask):
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    def forward(self, feat, max_decode=300):
        # 编码语言特征
        encoded_lang = self.encode_lang(feat)

        # 处理图数据
        scene_graph = feat['scene_graph_batch'].to(encoded_lang.device)
        task_graph = feat['task_graph_batch'].to(encoded_lang.device)
        batch_size = encoded_lang.size(0)

        # 编码图特征
        scene_emb = self.scene_encoder(scene_graph.x, scene_graph.edge_index)
        task_emb = self.task_encoder(task_graph.x, task_graph.edge_index)

        # 转换为密集表示
        scene_emb_dense, _ = self.to_dense_batch(scene_emb, scene_graph.batch, batch_size)
        task_emb_dense, _ = self.to_dense_batch(task_emb, task_graph.batch, batch_size)

        # 全局语言特征（作为图的全局特征）
        lang_global = encoded_lang.mean(dim=1)  # [batch, embed_dim]

        # 扩展全局特征
        scene_global = lang_global.unsqueeze(1).expand(-1, scene_emb_dense.size(1), -1)
        task_global = lang_global.unsqueeze(1).expand(-1, task_emb_dense.size(1), -1)

        # 融合全局特征
        scene_fused_input = torch.cat([scene_emb_dense, scene_global], dim=-1)
        scene_fused = self.scene_fusion_layer(scene_fused_input)
        task_fused_input = torch.cat([task_emb_dense, task_global], dim=-1)
        task_fused = self.task_fusion_layer(task_fused_input)

        # 节点匹配和融合
        matched_scene_feats, _ = self.node_matcher(task_fused, scene_fused)
        fused_task_emb = self.node_fusion(task_fused, matched_scene_feats)

        # 图级聚合
        scene_global_agg = scene_fused.mean(dim=1)
        task_global_agg = fused_task_emb.mean(dim=1)

        # 初始化图状态
        if self.scene_state is None:
            self.scene_state = torch.zeros(batch_size, self.embed_dim, device=encoded_lang.device)
        if self.task_state is None:
            self.task_state = torch.zeros(batch_size, self.embed_dim, device=encoded_lang.device)

        # 更新图状态
        self.scene_state = self.scene_gru(scene_global_agg, self.scene_state)
        self.task_state = self.task_gru(task_global_agg, self.task_state)

        # 融合图特征和语言特征
        lang_global = encoded_lang.mean(dim=1)
        graph_lang_feat = torch.cat([self.scene_state, self.task_state, lang_global], dim=-1)
        fused_feat = self.graph_lang_fusion(graph_lang_feat)

        # 将融合特征与每个时间步的语言特征结合
        fused_encoded_lang = encoded_lang + fused_feat.unsqueeze(1)

        # 处理视觉特征
        frames = self.vis_dropout(feat['frames'])

        # 解码
        res = self.dec(fused_encoded_lang, frames, max_decode=max_decode, gold=feat['action_low'])
        feat.update(res)

        return feat

    def to_dense_batch(self, x, batch, batch_size=None):
        if batch_size is None:
            batch_size = batch.max().item() + 1
        num_nodes = torch.bincount(batch)
        max_nodes = num_nodes.max().item()

        # 创建掩码
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            mask[i, :num_nodes[i]] = True

        # 创建密集批次
        dense_x = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        ptr = 0
        for i in range(batch_size):
            dense_x[i, :num_nodes[i]] = x[ptr:ptr + num_nodes[i]]
            ptr += num_nodes[i]

        return dense_x, mask

    def generate_attention_mask(self, seq_lengths, max_len):
        seq_lengths = seq_lengths.to(device)
        mask = torch.arange(max_len, device=device).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)
        return mask

    def encode_lang(self, feat):
        packed_lang_goal_instr = feat['lang_goal_instr']
        emb_lang_goal_instr, seq_lengths = pad_packed_sequence(packed_lang_goal_instr, batch_first=True)
        emb_lang_goal_instr = emb_lang_goal_instr.to(device)
        emb_lang_goal_instr = self.pos_encoder(emb_lang_goal_instr)

        attention_mask = self.generate_attention_mask(seq_lengths, emb_lang_goal_instr.size(1)).to(device)
        encoded_output = self.enc(emb_lang_goal_instr.transpose(0, 1), src_key_padding_mask=~attention_mask).transpose(
            0, 1)

        return encoded_output

    def reset(self):
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }
        # 重置图状态
        self.scene_state = None
        self.task_state = None

    def step(self, feat, prev_action=None):
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # 处理图数据
        if self.scene_state is None or self.task_state is None:
            scene_graph = feat['scene_graph_batch'].to(self.r_state['enc_lang'].device)
            task_graph = feat['task_graph_batch'].to(self.r_state['enc_lang'].device)
            batch_size = self.r_state['enc_lang'].size(0)

            # 编码图特征
            scene_emb = self.scene_encoder(scene_graph.x, scene_graph.edge_index)
            task_emb = self.task_encoder(task_graph.x, task_graph.edge_index)

            # 转换为密集表示
            scene_emb_dense, _ = self.to_dense_batch(scene_emb, scene_graph.batch, batch_size)
            task_emb_dense, _ = self.to_dense_batch(task_emb, task_graph.batch, batch_size)

            # 全局语言特征
            lang_global = self.r_state['enc_lang'].mean(dim=1)

            # 扩展全局特征
            scene_global = lang_global.unsqueeze(1).expand(-1, scene_emb_dense.size(1), -1)
            task_global = lang_global.unsqueeze(1).expand(-1, task_emb_dense.size(1), -1)

            # 融合全局特征
            scene_fused_input = torch.cat([scene_emb_dense, scene_global], dim=-1)
            scene_fused = self.scene_fusion_layer(scene_fused_input)
            task_fused_input = torch.cat([task_emb_dense, task_global], dim=-1)
            task_fused = self.task_fusion_layer(task_fused_input)

            # 节点匹配和融合
            matched_scene_feats, _ = self.node_matcher(task_fused, scene_fused)
            fused_task_emb = self.node_fusion(task_fused, matched_scene_feats)

            # 图级聚合
            scene_global_agg = scene_fused.mean(dim=1)
            task_global_agg = fused_task_emb.mean(dim=1)

            # 初始化图状态
            self.scene_state = scene_global_agg
            self.task_state = task_global_agg

            # 融合图特征和语言特征
            graph_lang_feat = torch.cat([self.scene_state, self.task_state, lang_global], dim=-1)
            self.fused_feat = self.graph_lang_fusion(graph_lang_feat)

        # 初始化嵌入和隐藏状态
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            # 融合图特征
            enc_lang_with_graph = self.r_state['enc_lang'] + self.fused_feat.unsqueeze(1)
            self.r_state['state_t'] = enc_lang_with_graph, torch.zeros_like(enc_lang_with_graph)

        # 前一个动作嵌入
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # 解码并保存嵌入和隐藏状态
        out_action_low, out_action_low_mask, state_t, *_ = self.dec.step(
            self.r_state['enc_lang'],
            feat['frames'][:, 0],
            e_t=e_t,
            state_tm1=self.r_state['state_t']
        )

        # 保存状态
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # 输出格式化
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)

        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # 移除填充标记
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]
            if clean_special_tokens:
                # 移除<>标记
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]
            # 索引到API动作
            words = self.vocab['action_low'].index2word(alow)
            # sigmoid预测到二进制掩码
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]
            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }
        return pred

    def embed_action(self, action):
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb

    def compute_loss(self, out, batch, feat):
        losses = dict()
        # GT和预测
        p_alow = out['out_action_low'].reshape(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].reshape(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # 动作损失
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # 掩码损失
        valid_idxs = valid.view(-1).nonzero().view(-1)
        total_elements = p_alow_mask.shape[0] * p_alow_mask.shape[1] * p_alow_mask.shape[2] * p_alow_mask.shape[3]
        valid_idxs = valid_idxs[valid_idxs < total_elements]

        # 重新计算 flat_p_alow_mask
        flat_p_alow_mask = p_alow_mask.view(-1, p_alow_mask.shape[2], p_alow_mask.shape[3])
        flat_alow_mask = torch.cat([torch.tensor(m, device=p_alow_mask.device) for m in feat['action_low_mask']], dim=0)

        # 确保形状一致
        if flat_p_alow_mask.size(0) > flat_alow_mask.size(0):
            flat_p_alow_mask = flat_p_alow_mask[:flat_alow_mask.size(0)]
        elif flat_p_alow_mask.size(0) < flat_alow_mask.size(0):
            flat_alow_mask = flat_alow_mask[:flat_p_alow_mask.size(0)]

        if len(valid_idxs) > 0 and flat_p_alow_mask.size(0) > 0 and flat_alow_mask.size(0) > 0:
            alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask[valid_idxs], flat_alow_mask[valid_idxs])
            losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt
        else:
            losses['action_low_mask'] = torch.tensor(0.0, device=p_alow.device)

        # 子目标完成损失
        if self.args.subgoal_aux_loss_wt > 0 and 'out_subgoal' in out:
            p_subgoal = out['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            if l_subgoal.size(1) > p_subgoal.size(1):
                l_subgoal = l_subgoal[:, :p_subgoal.size(1)]
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # 进度监控损失
        if self.args.pm_aux_loss_wt > 0 and 'out_progress' in out:
            p_progress = out['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            if l_progress.size(1) > p_progress.size(1):
                l_progress = l_progress[:, :p_progress.size(1)]
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses

    def focal_loss(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

    def weighted_mask_loss(self, pred_masks, gt_masks):
        gt_masks = gt_masks.squeeze(1).float()
        pred_masks = pred_masks.squeeze(1)
        return self.focal_loss(pred_masks, gt_masks)

    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def compute_metric(self, preds, data):
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v) / len(v) for k, v in m.items()}


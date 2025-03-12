import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv
from collections import Counter
from clip import load
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)
        corruption = "node_shuffle"
        self.corruption = getattr(self, "_%s" % corruption)

    def _node_shuffle(self, X):
        perm = torch.randperm(X.size(0))
        neg_X = X[perm]
        return neg_X

    def utterance_selector(self, key, context):
        '''
        Our
        :param key: (dim)
        :param context: (utts, dim)
        :return:(utts)
        '''
        s1 = torch.einsum("bu,u->b", context, key)/(1e-6 + torch.norm(context, dim=-1) * torch.norm(key, dim=-1, keepdim=True)) # 对应论文中公式(4):语义相关向量,list 类型
        return s1

    def utterance_selector_2(self, a, b): 
        """
            Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
            :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return (a_norm * b_norm).sum(-1)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:  # 每句话
                shanno = 0
                for token in utt:  # 每句话中的每个词
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)  # 信息熵，对应论文中的公式(6)
                entro.append(shanno)  # 这个对话的信息熵已添加
                entro.append(shanno)
                entro.append(shanno)
            dia_entro.append(torch.Tensor(entro))
        # batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)  # 用于将长度不一的序列（例如张量列表）填充到相同的长度
        # 使用 torch.cat 将张量列表连接成一个更大的张量
        batch_entro = torch.cat(dia_entro, dim=0)

        return batch_entro

    def get_batch_entropy_1(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)  # 信息熵，对应论文中的公式(6)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        # batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        batch_entro = torch.cat(dia_entro, dim=0)
        return batch_entro

    def forward(self, a, v, l, dia_len, qmask, epoch, Sentence):
        num_modality = len(self.modals)
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        '''Hyper GCN'''
        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)

        '''High GCN'''
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)
        # neg_gnn_out = self.corruption(gnn_out) # 负样本
        out2 = torch.cat([out,gnn_out], dim=1)
        # neg_out2 = torch.cat([neg_out,neg_gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
            # out2 = torch.cat([features, gnn_out], dim=-1)
            # neg_out2 = torch.cat([features, neg_out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        out2 = self.reverse_features(dia_len, out2)

        '''相似度矩阵：context filter s1'''
        utt_len = x1.size()[0]
        score1, scoreall = [], []
        for index in range(utt_len):
            key = x1[index, :]  # ":"表示选取该维度的所有元素，即选择该维度上的全部数据。 [512]
            s1 = self.utterance_selector(key, x1)  # 当前话语上下文相似度 [utt_len*3]
            s1 = s1.unsqueeze(1)
            s1 = s1.t()
            score1.append(s1)  # list类型：[utt_len*3]
        s1 = torch.cat(score1, dim=0)  # [utt_len*3, utt_len*3]
        edge_index1 = (s1 > 0).nonzero(as_tuple=False).t().contiguous()  # torch.Size([2, 4604671])
        # 计算归一化的注意力权重，可以使用 softmax 函数
        attention_weights = F.softmax(s1, dim=1)  # 在维度1上应用 softmax
        weighted_features = torch.matmul(attention_weights, x1)  # 注意力权重矩阵与特征矩阵相乘

        gnn_out_2 = weighted_features
        for kk in range(self.num_K):
            gnn_out_2 = gnn_out_2 + getattr(self, 'conv%d' % (kk + 1))(gnn_out_2, gnn_edge_index)

        out3 = torch.cat([out, gnn_out_2], dim=1) 
        if self.use_residue:
            out3 = torch.cat([features, out3], dim=-1)

        out3 = self.reverse_features(dia_len, out3)

        '''信息熵: 用Sentence来进行计算'''
        device = gnn_features.device
        # print(len(Sentence)) # BN大小
        if num_modality == 3:
            s2 = self.get_batch_entropy(Sentence).to(device)
            # 计算归一化的注意力权重，可以使用 softmax 函数
            attention_weights = F.softmax(s2, dim=0)  # 在维度0上应用 softmax
            # 将注意力权重应用于特征向量
            weighted_features = attention_weights.unsqueeze(-1) * x1  
        elif num_modality == 1:
            s2 = self.get_batch_entropy_1(Sentence).to(device)
            # 计算归一化的注意力权重，可以使用 softmax 函数
            attention_weights = F.softmax(s2, dim=0)  # 在维度0上应用 softmax
            # 将注意力权重应用于特征向量
            weighted_features = attention_weights.unsqueeze(-1) * x1
        gnn_out_3 = weighted_features
        for kk in range(self.num_K):
            gnn_out_3 = gnn_out_3 + getattr(self, 'conv%d' % (kk + 1))(gnn_out_3, gnn_edge_index)
        out4 = torch.cat([weighted_features, gnn_out_3], dim=1) 

        if self.use_residue:
            out4 = torch.cat([features, out4], dim=-1)
        out4 = self.reverse_features(dia_len, out4)

        return out1, out2, out3, out4
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i + 3):
                if _ < 3:
                    index2 = index2 + [edge_count] * i
                else:
                    index2 = index2 + [edge_count] * 3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i

            Gnodes = []
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_, 2))
                tmp = tmp + perm
            batch = batch + [batch_count] * i * 3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1] * i + [0] * 3

            node_count = node_count + i * num_modality

        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0, max_node + 1, 1).repeat_interleave(2).view(1, -1),
                               torch.arange(max_edge + 1, max_edge + 1 + max_node + 1, 1).repeat_interleave(2).view(1,
                                                                                                                    -1)],
                              dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).cuda()


        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index = index + list(permutations(nodes_l, 2)) + list(permutations(nodes_a, 2)) + list(
                permutations(nodes_v, 2))
            Gnodes = []
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp + list(permutations(_, 2))
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i
            node_count = node_count + i * num_modality
        edge_index = torch.cat([torch.LongTensor(index).T, torch.LongTensor(tmp).T], 1).cuda()

        return edge_index, features

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

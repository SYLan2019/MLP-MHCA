import numpy as np
import torch
from torch import nn
from torch.nn import init


class CrossDotProductAttention(nn.Module):
    '''
    Cross dot-product attention
    '''

    def __init__(self, d_model, d_qk, h=1, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_qk: Dimensionality of queries and keys
        :param h: Number of heads
        '''
        super(CrossDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_qk)
        self.fc_k = nn.Linear(d_model, h * d_qk)
        self.dropout = nn.Dropout(dropout)
        self.adjust = nn.Linear(2, 1)

        self.d_model = d_model
        self.d_k = d_qk
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        att_max, _ = torch.max(att, dim=3, keepdim=True)
        att_mean = torch.mean(att, dim=3, keepdim=True)
        att = torch.cat([att_max, att_mean], 3)
        att = torch.softmax(self.adjust(att), -1).permute(0, 2, 1, 3).view(b_s, nq, self.h)
        out = att*queries

        return queries + out


class ResponseMap(nn.Module):
    def __init__(self, d_model, d_qk, h=64, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_qk: Dimensionality of queries and keys
        :param h: Number of heads
        '''
        super(ResponseMap, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_qk)
        self.fc_k = nn.Linear(d_model, h * d_qk)
        self.dropout = nn.Dropout(dropout)
        self.adjust = nn.Linear(2, 1)

        self.d_model = d_model
        self.d_k = d_qk
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, queries, keys):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        att_max, _ = torch.max(att, dim=3, keepdim=True)
        att_mean = torch.mean(att, dim=3, keepdim=True)
        out = torch.cat([att_max, att_mean], 3)
        out = self.adjust(out).permute(0, 2, 1, 3).view(b_s, nq, self.h)

        return out

# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 9:46 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : pointer_net.py
# @Software: PyCharm

import torch
import math
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, ref):
        query = query.unsqueeze(2)
        logits = torch.bmm(ref, query).squeeze(2)
        ref = ref.permute(0, 2, 1)
        return ref, logits


class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()
        self.args = args

        self.hidden_size = self.args.hidden_size

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        # self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(self.hidden_size, nhead=4, dim_feedforward=self.hidden_size, \
                                             batch_first=True, dropout=0), num_layers=1
        )
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.pointer = Attention()

        self.decoder_init_input = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.decoder_init_input.data.uniform_(-(1. / math.sqrt(self.hidden_size)), 1. / math.sqrt(self.hidden_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[list(range(batch_size)), idxs] = 1
            logits[clone_mask.bool()] = torch.tensor(-1e12)
        return logits, clone_mask

    def forward(self, inputs):
        B, S, F = inputs.size()

        features = self.embedding(inputs)
        encoder_out = self.encoder(features)
        hidden = torch.max(encoder_out, dim=1)[0].unsqueeze(0)
        cell = torch.max(encoder_out, dim=1)[0].unsqueeze(0)

        decoder_input = self.decoder_init_input.repeat(B, 1)
        idx = None
        mask = torch.zeros(B, S).byte()

        actions = []
        action_idx = []
        action_prob = []
        for i in range(S):
            decoder_out, (hidden, cell) = self.decoder(decoder_input.unsqueeze(1), (hidden, cell))
            _, logits = self.pointer(decoder_out.squeeze(1), encoder_out)
            logits, mask = self.apply_mask_to_logits(logits, mask, idx)
            probs = torch.nn.functional.softmax(logits, dim=1)  # [B, S]
            idx = probs.multinomial(1).squeeze(1)
            decoder_input = features[list(range(B)), idx, :]  # [B, F]
            actions.append(inputs[list(range(B)), idx])
            action_idx.append(idx)
            action_prob.append(probs[list(range(B)), idx])

        return action_prob, actions, action_idx
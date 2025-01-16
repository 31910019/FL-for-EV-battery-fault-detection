import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from DL_model import DNN, Transformer_ori, LSTM, biLSTM

class LSTM_AE(nn.Module):
    def __init__(self,args):
        super(LSTM_AE,self).__init__()
        self.args_enc = copy.deepcopy(args)
        self.args_enc.m = 2
        self.args_dec = copy.deepcopy(args)
        self.args_dec.m = self.args_enc.output_size
        self.args_dec.output_size = 1
        self.original_len = args.window_size
        # self.output_size = args.output_size
        # self.hidden_size = 48
        self.emb_size = 100
        #[batchsize, feature_dim, seq_len]
        # print(self.args_enc.m)
        # print(self.args_dec.output_size)
        self.encoder = biLSTM(self.args_enc)
        self.time_fc_enc = nn.Linear(self.original_len, self.emb_size)
        self.time_fc_dec = nn.Linear(self.emb_size, self.original_len)
        self.decoder = biLSTM(self.args_dec)

    def forward(self,x):
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        x = self.encoder(x)
        # print(x.shape)
        x = x.transpose(2,1)
        emb = self.time_fc_enc(x)
        y = self.time_fc_dec(emb)
        y = y.transpose(2,1)
        y = self.decoder(y)
        # print(x.shape)
        # x = torch.squeeze(x,dim=2)
        return y


class transformer_AE(nn.Module):
    def __init__(self,args):
        super(transformer_AE,self).__init__()
        self.args_enc = copy.deepcopy(args)
        # self.args_enc.m = 2
        self.args_dec = copy.deepcopy(args)
        self.args_dec.m = self.args_enc.output_size + self.args_enc.m
        self.args_dec.output_size = 1
        self.original_len = args.window_size
        # self.output_size = args.output_size
        # self.hidden_size = 48
        self.emb_size = 50
        #[batchsize, feature_dim, seq_len]
        # print(self.args_enc.m)
        # print(self.args_dec.output_size)
        self.encoder = Transformer_ori(self.args_enc)
        self.time_fc_enc = nn.Linear(self.original_len, self.emb_size)
        self.time_fc_dec = nn.Linear(self.emb_size, self.original_len)
        self.decoder = Transformer_ori(self.args_dec)

    def forward(self,x):
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        x_ori = copy.deepcopy(x)
        x = self.encoder(x)
        # print(x.shape)
        x = x.transpose(2,1)
        emb = self.time_fc_enc(x)
        y = self.time_fc_dec(emb)
        y = y.transpose(2,1)
        y = torch.concat([y,x_ori], axis=2)
        # print(y.shape)
        y = self.decoder(y)
        # print(x.shape)
        # x = torch.squeeze(x,dim=2)
        return y
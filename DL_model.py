import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, args, is_sigmoid=0):
        super().__init__()
        self.args = args
        self.is_sigmoid = is_sigmoid
        self.original_len = args.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = args.drop_out

        self.hidden = 4
        self.heads = 2
        self.n_layers = 1
        self.output_size = args.output_size

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden,
                               kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)
        if self.is_sigmoid:
            self.act = nn.Sigmoid()
        self.truncated_normal_init()
        print('self.output_size ',self.output_size )

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):    #torch.Size([128, 480])
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)     #torch.Size([128, 480, 1])
        # print('ssss',x.shape)
        if self.is_sigmoid:
            x = self.act(x)
        return x


class Transformer_ori(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size
        self.latent_len = self.original_len
        self.dropout_rate = args.drop_out

        self.hidden = 16
        self.heads = 16
        self.n_layers = 2
        self.output_size = args.output_size

        # self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden,
        #                        kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        # self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.linear0 = nn.Linear(self.args.m, self.hidden)
        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden)
        # self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        # self.deconv = nn.ConvTranspose1d(
        #     in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)
        self.act = nn.Sigmoid()

        self.truncated_normal_init()
        # print('self.output_size ',self.output_size )

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):    #torch.Size([128, 480])
        x_token = self.linear0(sequence)
        # print(sequence.shape)
        # print(self.position(sequence).shape)
        embedding = x_token + self.position(sequence)

        x = self.dropout(embedding)
        # print(x.shape)
        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x_enc = copy.deepcopy(x)
        x_res = self.linear1(x)
        x_res = self.linear2(x_res)     #torch.Size([128, 480, 1])
        # x = self.act(x)
        # print('ssss',x.shape)
        return x_res, x


class DNN(nn.Module):
    def __init__(self, args=None, input_size=480, output_size=480, num_layers=3, num_neurals=[128, 256, 128], dropout_rate=0.1, final_act=False):
        super().__init__()
        self.args = args

        self.num_layers = num_layers
        self.num_neurals = num_neurals
        self.dropout_rate = dropout_rate
        # self.dropout = nn.Dropout(self.dropout_rate)
        self.input_size = self.args.window_size
        self.output_size = self.args.window_size
        self.activation = nn.ReLU()
        self.num_neurals = [self.input_size] + self.num_neurals
        self.final_act = final_act
        assert num_layers == len(num_neurals)

        module_list = []
        for i in range(self.num_layers):
            module_list.append(nn.Linear(self.num_neurals[i], self.num_neurals[i+1]))
            module_list.append(self.activation)
            module_list.append(nn.Dropout(self.dropout_rate))

        module_list.append(nn.Linear(self.num_neurals[-1], self.output_size))
        if self.final_act == True:
            module_list.append(self.activation)

        self.net = nn.Sequential(*module_list)

    def forward(self, sequence):

        return self.net(sequence)


class CNN(nn.Module):
    def __init__(self,args):
        super(CNN,self).__init__()
        self.args = args
        self.original_len = args.window_size
        self.output_size = args.output_size
        #[batchsize, feature_dim, seq_len]
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=5,stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=5,stride=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256,out_channels=256,kernel_size=5,stride=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=5,stride=2)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024,self.original_len)
        self.drp = nn.Dropout(0.3)

    def forward(self,x):
        # print(x.shape)
        x = torch.unsqueeze(x,dim=1)
        # print(x.shape)
        x = self.bn1(self.conv1(x))
        # print('x')
        # print(x.shape)
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        # print(x.shape)
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = self.bn4(self.conv4(x))
        x = F.relu(x)
        x = self.bn5(self.conv5(x))
        x = F.relu(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drp(x)
        x = F.relu(self.fc1(x))
        x = self.drp(x)
        x = self.fc2(x)
        x = torch.unsqueeze(x,dim=2)
        return x


class LSTM(nn.Module):
    def __init__(self,args):
        super(LSTM,self).__init__()
        self.args = args
        self.original_len = args.window_size
        self.output_size = args.output_size
        self.hidden_size = 16
        #[batchsize, feature_dim, seq_len]
        self.lstm = nn.LSTM(input_size=args.m,hidden_size=self.hidden_size,num_layers=3,batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, 1)
        # self.fc2 = nn.Linear(1024,self.original_len)
        self.drp = nn.Dropout(0.1)
        # self.act = nn.Sigmoid()

    def forward(self,x):
        # print(x.shape)
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        # print(x.shape)
        x, _ = self.lstm(x)
        x = self.drp(x)
        x = self.fc1(x)
        # x = self.act(x)
        # x = torch.squeeze(x,dim=2)
        return x

class biLSTM(nn.Module):
    def __init__(self,args):
        super(biLSTM,self).__init__()
        self.args = args
        self.original_len = args.window_size
        self.output_size = args.output_size
        self.hidden_size = 16
        #[batchsize, feature_dim, seq_len]
        self.lstm = nn.LSTM(input_size=args.m,hidden_size=self.hidden_size,num_layers=3,batch_first=True,bidirectional=True)

        self.fc1 = nn.Linear(2*self.hidden_size, self.output_size)
        # self.fc2 = nn.Linear(1024,self.original_len)
        self.drp = nn.Dropout(0.1)
        # self.act = nn.Sigmoid()

    def forward(self,x):
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        x, _ = self.lstm(x)
        x = self.drp(x)
        x = self.fc1(x)
        # x = self.act(x)
        # x = torch.squeeze(x,dim=2)
        return x

class GRU(nn.Module):
    def __init__(self,args):
        super(GRU,self).__init__()
        self.args = args
        self.original_len = args.window_size
        self.output_size = args.output_size
        self.hidden_size = 16
        #[batchsize, feature_dim, seq_len]
        self.gru = nn.GRU(input_size=args.m,hidden_size=self.hidden_size,num_layers=3,batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, 1)
        # self.fc2 = nn.Linear(1024,self.original_len)
        self.drp = nn.Dropout(0.1)
        # self.act = nn.Sigmoid()

    def forward(self,x):
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        x, _ = self.gru(x)
        x = self.drp(x)
        x = self.fc1(x)
        # x = self.act(x)
        # x = torch.squeeze(x,dim=2)
        return x

class biGRU(nn.Module):
    def __init__(self,args):
        super(biGRU,self).__init__()
        self.args = args
        self.original_len = args.window_size
        self.output_size = args.output_size
        self.hidden_size = 48
        #[batchsize, feature_dim, seq_len]
        self.gru = nn.GRU(input_size=args.m,hidden_size=self.hidden_size,num_layers=5,batch_first=True,bidirectional=True)

        self.fc1 = nn.Linear(2*self.hidden_size, self.output_size)
        # self.fc2 = nn.Linear(1024,self.original_len)
        self.drp = nn.Dropout(0.1)
        # self.act = nn.Sigmoid()

    def forward(self,x):
        # print(x.shape)
        # x = torch.unsqueeze(x,dim=2)
        x, _ = self.gru(x)
        x = self.drp(x)
        x = self.fc1(x)
        # x = self.act(x)
        # print(x.shape)
        # x = torch.squeeze(x,dim=2)
        return x


from torch.nn.utils import weight_norm
from torch.autograd import Variable
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # print('n')
        # print(n_inputs)
        # print(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        #self.conv1.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        #self.conv2.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        if self.downsample is not None:
            #self.downsample.weight.data.normal_(0, 0.01)
            nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))

    def forward(self, x):

        # print('start')
        x_ori = x
        # print(x.shape)
        # net = self.net(x)
        x = self.conv1(x)
        # print(x.shape)
        x = self.chomp1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.chomp2(x)
        # print(x.shape)
        res = x_ori if self.downsample is None else self.downsample(x_ori)
        return self.relu(x + res)


class TCN(nn.Module):
    def __init__(self, args, num_channels=[16,16,16], kernel_size=2, dropout=0.2, max_length=200, attention=False):
        super(TCN, self).__init__()
        self.args = args
        num_inputs = args.m
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):

            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            if attention == True:
                layers += [AttentionBlock(max_length, max_length, max_length)]


        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):

        x = torch.unsqueeze(x, dim=1)
        x = self.network(x)
        x = self.fc(x.transpose(1,2))
        # print(x.shape)
        return x

class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)
      The input of the AttentionBlock is `BxTxD` where `B` is the input
      minibatch size, `T` is the length of the sequence `D` is the dimensions of
      each feature.
      The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
      attention values.
      Arguments:
          dims (int): the number of dimensions (or channels) of each element in
              the input sequence
          k_size (int): the size of the attention keys
          v_size (int): the size of the attention values
          seq_len (int): the length of the input and output sequences
    """
    def __init__(self, dims, k_size, v_size, seq_len=None):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        # Use numpy triu because you can't do 3D triu with PyTorch
        # TODO: using float32 here might break for non FloatTensor inputs.
        # Should update this later to use numpy/PyTorch types of the input.
        mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
        mask = torch.from_numpy(mask).cuda()
        # do masked_fill_ on data rather than Variable because PyTorch doesn't
        # support masked_fill_ w/-inf directly on Variables for some reason.
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits, dim=1) / self.sqrt_k
        read = torch.bmm(probs, values)

        return minibatch + read


class MMOE(nn.Module):

    def __init__(self, args, input_size=480, num_expert=3, expert_activation=None, num_task=2):
        super(MMOE, self).__init__()

        self.args = args
        self.expert_activation = expert_activation
        self.num_task = num_task
        self.input_size = self.args.window_size
        self.num_expert = num_expert

        self.expert_1 = TCN(args)
        self.expert_2 = DNN(args)
        self.expert_3 = TCN(args)

        self.gates = nn.Linear(self.input_size,self.num_expert*self.num_task)
        # self.gate_activation = nn.Sigmoid()

        # esmm ctr和ctcvr独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}'.format(i + 1), Transformer(self.args))

    def forward(self, x):
        # print(x.dtype)
        expert_outs = []
        expert_outs.append(self.expert_1(x))
        expert_outs.append(self.expert_2(x))
        expert_outs.append(self.expert_3(x))
        # print(expert_outs[0].shape)
        # print(expert_outs[2].shape)
        gate_out = self.gates(x)
        gate_out = gate_out.reshape([-1, self.num_expert, self.num_task])
        # gate_out = nn.Softmax(dim=1)(gate_out)
        # print(gate_out.shape)
        gates_out = []
        for idx in range(self.num_task):
            # batch * num_experts
            gate_out_r = nn.Softmax(dim=1)(gate_out[:,:,idx])
            gates_out.append(gate_out_r)

        outs = []
        for gate_output in gates_out:
            # gate_out  # batch * num_experts
            weighted_expert_output = torch.zeros(expert_outs[0].squeeze().shape).to(self.args.device)
            # gate_output = gate_output.to(self.args.device)
            for k,expert_out in enumerate(expert_outs):
                # expert_out  # batch * hidden_size
                # print('cao')
                # print(gate_output[:,k].reshape([-1,1]).shape)
                # print(expert_outs[0].shape)
                # print(expert_outs[2].shape)
                # print(expert_out.squeeze().shape)
                # print(weighted_expert_output.shape)
                weighted_expert_output += expert_out.squeeze() * gate_output[:,k].reshape([-1,1])  # batch * hidden_size
            outs.append(weighted_expert_output)  # batch * hidden_size

        # task tower
        task_outputs = []
        for i in range(self.num_task):
            x = outs[i]
            mod = getattr(self, 'task_{}'.format(i + 1))
            # print(x.dtype)
            x = mod(x.to(torch.float32))
            task_outputs.append(x)

        return task_outputs



class MMOE_UC(nn.Module):
    def __init__(self, args, input_size=480, num_expert=3, expert_activation=None, num_task=17):
        super(MMOE_UC, self).__init__()

        self.args = args
        self.expert_activation = expert_activation
        self.num_task = num_task
        self.window_size = self.args.window_size
        self.input_size = args.m
        self.num_expert = num_expert
        self.args_trans = copy.deepcopy(self.args)
        self.args_trans.m = 1
        self.args_cost = copy.deepcopy(self.args)
        self.args_cost.window_size = 4
        self.args_cost.m = 1

        self.expert_1 = LSTM(args)
        self.expert_2 = GRU(args)
        self.expert_3 = Transformer_ori(args)

        self.gates = nn.Linear(self.input_size,self.num_expert*self.num_task)
        # self.gate_activation = nn.Sigmoid()

        for i in range(self.num_task-1):  # 去掉最后一个cost 的gate，因为维度不同
            setattr(self, 'task_{}'.format(i + 1), Transformer_ori(self.args_trans))
        setattr(self, 'task_{}'.format(self.num_task), nn.Linear(self.window_size, self.args_cost.window_size))

    def forward(self, x):
        # print(x.dtype)
        expert_outs = []
        expert_outs.append(self.expert_1(x))
        expert_outs.append(self.expert_2(x))
        expert_outs.append(self.expert_3(x))
        # print(expert_outs[0].shape)
        # print(expert_outs[2].shape)
        gate_out = self.gates(x)
        # print(x.shape)
        # print(gate_out.shape)
        gate_out = gate_out.reshape([-1, self.window_size, self.num_expert, self.num_task])
        # gate_out = nn.Softmax(dim=1)(gate_out)
        # print(gate_out.shape)
        gates_out = []
        for idx in range(self.num_task):
            # batch * num_experts
            # print(gate_out[:,:,:,idx].shape)
            gate_out_r = nn.Softmax(dim=1)(gate_out[:,:,:,idx])
            # print('r')
            # print(gate_out_r.shape)
            gates_out.append(gate_out_r)

        outs = []
        for gate_output in gates_out:
            # gate_out  # batch * num_experts
            weighted_expert_output = torch.zeros(expert_outs[0].squeeze(dim=2).shape).to(self.args.device)
            # gate_output = gate_output.to(self.args.device)
            for k,expert_out in enumerate(expert_outs):
                # expert_out  # batch * hidden_size
                # print('x')
                # print(gate_output[:,:,k].shape)
                # print(expert_out.squeeze(dim=2).shape)
                # print(weighted_expert_output.shape)
                # print(expert_outs[2].shape)
                # print(expert_out.squeeze().shape)
                # print(weighted_expert_output.shape)
                weighted_expert_output += expert_out.squeeze() * gate_output[:,:,k]  # batch * hidden_size
            outs.append(weighted_expert_output)  # batch * hidden_size

        # task tower
        task_outputs = []
        for i in range(self.num_task-1):
            x = outs[i].unsqueeze(dim=2)
            mod = getattr(self, 'task_{}'.format(i + 1))
            # print(x.dtype)
            # print(x.shape)
            x = mod(x.to(torch.float32))
            # print(x.shape)
            task_outputs.append(x)
        mod = getattr(self, 'task_{}'.format(self.num_task))
        # print(x.dtype)
        # print(outs[-1].shape)
        x = mod(outs[-1].to(torch.float32))
        # x = torch.sigmoid(x)
        task_outputs.append(x)

        # print(task_outputs[-1])

        return task_outputs



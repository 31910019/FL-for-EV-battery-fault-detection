import time

import scipy.io as scio
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, TensorDataset, DataLoader
from DL_detection_model import LSTM_AE, transformer_AE, Transformer_ori
from DL_model import TCN
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from experiments.pfedhn.models import Hyper
from utils.utils import time_series_EV_charger_dataset, FL_time_series_EV_charger_dataset_2
import random
from collections import OrderedDict, defaultdict
from collections import Counter
import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

Tn = 20313
Tf = 862
selection_ratio = 2*Tf/Tn


train_data = []
test_data = []
train_label = []
test_label = []
file_path = 'processed_data_longer_than_30.xlsx'

n_slides = 8
# writer_1 = pd.ExcelWriter('processed_data_train.xlsx', engine='xlsxwriter')
# writer_2 = pd.ExcelWriter('processed_data_test.xlsx', engine='xlsxwriter')

start = 0
count = []
for k in range(n_slides):
    df = pd.read_excel(file_path, sheet_name='Sheet'+str(k+1))
    normalf_t = df[df['label'] == 0]
    faultf_t = df[df['label'] == 1]
    # normalf_t = normalf[normalf['class_judge'].isin(point_list[k+1])]
    normal_split = start + int(selection_ratio*normalf_t.shape[0])
    normal_split_train = int(2*normal_split/4)
    while normal_split_train < normal_split:
        if normalf_t.iloc[start+normal_split_train, 5] != normalf_t.iloc[start+normal_split_train + 1, 5]:
            # normal_split_train -= 1
            break
        normal_split_train -= 1
    normalf_train = normalf_t.iloc[start:start+normal_split_train]
    normalf_test = normalf_t.iloc[start+normal_split_train:start+normal_split]

    # normalf_tt = normalf_t.iloc[start:start+normal_split]
    # for kkt in [3,6,4,5,7]:
    #     ap = dict(Counter(normalf_tt[normalf_tt['types'] == kkt].transaction_id))
    #     print(sum(1 for value in ap.values() if value > 30))


    # faultf_t = faultf[faultf['class_judge'].isin(point_list[k+1])]
    # print(faultf_t.shape)
    fault_split_train = int(2*faultf_t.shape[0]/4)
    while fault_split_train < faultf_t.shape[0]:
        if faultf_t.iloc[fault_split_train, 5] != faultf_t.iloc[fault_split_train + 1, 5]:
            # split_test += 1
            break
        fault_split_train -= 1
    faultf_train = faultf_t.iloc[:fault_split_train]
    faultf_test = faultf_t.iloc[fault_split_train:]

    dataf_train = normalf_train.append(faultf_train)
    dataf_train = dataf_train[['id','transaction_id','begin_time','end_time','total_charging_kwh','total_charging_min','current_soc','current_energy_meter_value','chargingv','charginga','out_power','charging_gun_temperature1','charging_gun_temperature2', 'types', 'class_judge', 'label']]
    dataf_test = normalf_test.append(faultf_test)
    dataf_test = dataf_test[['id','transaction_id','begin_time','end_time','total_charging_kwh','total_charging_min','current_soc','current_energy_meter_value','chargingv','charginga','out_power','charging_gun_temperature1','charging_gun_temperature2', 'types', 'class_judge', 'label']]

    # dataf_train.to_excel(writer_1, sheet_name='Sheet'+str(k+1), index=False)
    # dataf_test.to_excel(writer_2, sheet_name='Sheet'+str(k+1), index=False)
    # print(dataf_train["transaction_id"].nunique())
    # # print(dataf_test["transaction_id"].nunique())
    # first_rows_train = dataf_train.groupby('transaction_id').first().reset_index()
    # count_per_id_train = dataf_train['transaction_id'].value_counts().reset_index()
    # count_per_id_train.columns = ['transaction_id', 'count']
    # result_train = pd.merge(first_rows_train, count_per_id_train, on='transaction_id')
    # result_train = result_train[result_train["count"]>30]
    # first_rows_test = dataf_test.groupby('transaction_id').first().reset_index()
    # count_per_id_test = dataf_test['transaction_id'].value_counts().reset_index()
    # count_per_id_test.columns = ['transaction_id', 'count']
    # result_test = pd.merge(first_rows_test, count_per_id_test, on='transaction_id')
    # result_test = result_test[result_test["count"]>30]
    # print(sum(result_train['label']) + sum(result_test['label']))
    # print(result_train.shape[0]+result_test.shape[0])

    # count.append(dataf_train["transaction_id"].nunique())
    # count.append(dataf_test["transaction_id"].nunique())
    train_data.append(dataf_train.values)
    train_label.append(dataf_train['label'].values)
    test_data.append(dataf_test.values)
    test_label.append(dataf_test['label'].values)
    print(train_data[-1].shape)
    print(test_data[-1].shape)



n_slides = 8

class args():
    def __init__(self):
        self.win_size = 30
        self.window_size = self.win_size
        self.batch_size = 300
        self.output_size = 1
        self.lr = 1e-5
        self.inner_lr = 5e-3
        self.m = 3
        # self.contamination_rate = contamination
        self.drop_out = 0.2
        self.n_sides = 6   # only the first six was used
        self.embedding_dim = 346
        self.num_round = 1
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

args = args()

train_set_list = []
test_set_list = []
train_loader_list = []
test_loader_list = []

for k in range(n_slides):
    train_set = time_series_EV_charger_dataset(args, train_data[k], train_label[k])
    test_set = time_series_EV_charger_dataset(args, test_data[k], test_label[k])
    train_set.normalize()
    x_mean, x_std = train_set.get_mean_std()
    test_set.normalize()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    train_set_list.append(train_set)
    test_set_list.append(test_set)
    train_loader_list.append(train_loader)
    test_loader_list.append(test_loader)

# from DL_detection_model import LSTM_AE
model_list = []
for k in range(args.n_sides):
    model_list.append(Transformer_ori(args).to(args.device))

hyper_model = Transformer_ori(args).to(args.device)

def train(args, hyper_model, model_list, train_loader, test_loader):
    num_epochs = 600
    CE = torch.nn.BCEWithLogitsLoss()
    print("======================TRAIN MODE======================")
    acc_list = []
    f1_list = []
    acc_t_list = []
    f1_t_list = []
    save_score_max = 0
    save_score_epoch = 0
    optimizer = torch.optim.Adam(params=hyper_model.parameters(), lr=args.lr)

    for epoch in range(num_epochs):
        iter_count = 0
        loss_list = []
        # label_list_train = []
        # dis_list_train = []
        # epoch_time = time.time()
        pred_list_train = []
        label_list_train = []
        # for node_id in range(args.n_sides):
        #     dis_list_train.append([])
        node_id_list = []
        node_id = random.choice(range(args.n_sides))
        node_id_list.append(node_id)
        node_id = random.choice(range(args.n_sides))
        node_id_list.append(node_id)
        # hyper_model.train()
        weights = hyper_model.state_dict()



        for node_id in node_id_list:
            for i, (input_data, labels) in enumerate(tqdm(train_loader_list[node_id])):
                # label_list_train.extend((torch.sum(labels, axis=1)>0).cpu().detach().numpy().tolist())
                model_list[node_id].load_state_dict(weights)
                model_list[node_id].train()
                inner_optim = torch.optim.Adam(params=model_list[node_id].parameters(), lr=args.inner_lr)
                inner_optim.zero_grad()
                iter_count += 1
                input_data = input_data.float().to(args.device)
                output, _ = model_list[node_id](input_data[:,:,4:7])

                list_weights = list(weights.values())
                temp = list(model_list[node_id].parameters())
                weights_square = 0
                for ai, aj in zip(list_weights, temp):
                    weights_square += torch.sum(ai - aj)
                # print(labels.shape)
                # print(output.shape)
                loss = CE(output.squeeze(), labels.to('cuda:0')) + 0.2*weights_square
                loss.backward()
                loss_list.append(loss.cpu().detach().numpy())
                # torch.nn.utils.clip_grad_norm_(model_list[node_id].parameters(), 50)
                inner_optim.step()

                pred_list_train.extend((output>=0.5).cpu().detach().numpy())
                label_list_train.extend(labels.cpu().detach().numpy())

        avg_state = model_list[node_id_list[0]].state_dict()
                # calculating delta theta
        for kk in node_id_list[1:]:
            temp_state = model_list[kk].state_dict()
            for kj in avg_state.keys():
                avg_state[kj] += temp_state[kj]

        for kj in avg_state.keys():
            avg_state[kj] = avg_state[kj]/len(node_id_list)

        hyper_model.load_state_dict(avg_state)


        iter_count = 0

        dis_list_test = []
        label_list_test = []
        test_loss_list = []
        pred_list_test = []
        for node_id in range(args.n_sides):
            dis_list_test.append([])
            label_list_test.append([])
            test_loss_list.append([])
            pred_list_test.append([])

        for node_id in range(args.n_sides):
            model_list[node_id].load_state_dict(weights)
            model_list[node_id].eval()
            for i, (input_data, labels) in enumerate(test_loader_list[node_id]):
                input_data = input_data.float().to(args.device)
                iter_count += 1
                weights = hyper_model.state_dict()
                model_list[node_id].load_state_dict(weights)
                output, _ = model_list[node_id](input_data[:, :, 4:7])
                # output = output.squeeze()
                loss_test = CE(output.squeeze(), labels.to('cuda:0'))
                test_loss_list[node_id].extend([loss_test.cpu().detach().numpy().tolist()])
                pred_list_test[node_id].extend((output >= 0.5).cpu().detach().numpy())
                label_list_test[node_id].extend(labels.cpu().detach().numpy().tolist())

        acc_t_list = []
        f_score_t_list = []
        recall_list = []
        for node_id in range(args.n_sides):
            precision, recall, _, support = precision_recall_fscore_support(
                np.array(label_list_test[node_id]).reshape([-1]),
                np.array(pred_list_test[node_id]).squeeze().reshape([-1]))
            acc_t = accuracy_score(np.array(label_list_test[node_id]).reshape([-1]),
                                   np.array(pred_list_test[node_id]).squeeze().reshape([-1]))
            # print(acc_t)
            f_score = f1_score(np.array(label_list_test[node_id]).reshape([-1]),
                               np.array(pred_list_test[node_id]).squeeze().reshape([-1]))
            acc_t_list.append(acc_t)
            f_score_t_list.append(f_score)
            recall_list.append(recall[1])

        train_loss = np.average(loss_list)
        test_loss_list_merge = [item for sublist in test_loss_list for item in sublist]
        test_loss = np.average(test_loss_list_merge)

        print("epoch {0}, train_loss: {1}, test_loss: {2}, acc: {3}, f1: {4}, recall:{5}".format(
            epoch, train_loss, test_loss, acc_t_list, f_score_t_list, recall_list))

        # save_score = np.mean(sum(acc_t_list)) / len(acc_t_list) + 0.5 * sum(f_score_t_list) / len(f_score_t_list)
        # if save_score > save_score_max:
        #     model_path = 'best_model.pth'
        #     torch.save(hyper_model.state_dict(), model_path)
        #     save_score_max = save_score
        #     print('save model with save score {0}'.format(save_score))
        #     save_score_epoch = epoch - 1

    return pred_list_test, label_list_test, np.array(acc_list), np.array(f1_list), save_score_epoch

pred_test, label_list_test, acc_list, f1_list, save_score_epoch = train(args, hyper_model, model_list, train_loader,
                                                                            test_loader)

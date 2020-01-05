import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import torch.nn.functional as F
from Dataset import MyDataset
import torch
from torch import nn
import copy
from resnet import resnet18

trn_path = 'hy_round1_train_20200102'
test_path = 'hy_round1_testA_20200102'
trn_batchsize = 64
trn_figurepath = 'train'
test_figurepath = 'test'
max_epoch = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)


def get_label(path):
    labels = []
    for file in tqdm.tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        tmp_label = df['type'][0]
        if tmp_label == '拖网':
            tmp_label = 0
        elif tmp_label == '刺网':
            tmp_label = 1
        else:
            tmp_label = 2
        labels.append(tmp_label) 
        
    return labels


def calc_f1(target, output):
    y_true = target.cpu().detach().numpy()
    # y_true = np.argmax(y_true, axis=1)
    y_pre = output.cpu().detach().numpy()
    y_pre = np.argmax(y_pre, axis=1)
    return f1_score(y_true, y_pre, average='macro')


def train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, batch_size):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    tq = tqdm.tqdm(total=len(train_dataloader)*batch_size)
    tq.set_description('folds: %d, epoch %d, lr %.4f, best_f:%.4f' % (fold_+1, epoch, lr, best_f1))
    for i, (inputs, target) in enumerate(train_dataloader):
#        print(1)
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        output = F.softmax(output, dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = calc_f1(target, output)
        f1_meter += f1
        tq.update(batch_size)
        tq.set_postfix(loss="%.4f   f1:%.3f" % (loss.item(), f1))
    tq.close()
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, batch_size):
    model.eval()
    loss_meter, it_count = 0, 0
    with torch.no_grad():
        if torch.cuda.is_available():
            label_all = torch.Tensor().long().cuda()
            pred_all = torch.Tensor().cuda()
        else:
            label_all = torch.Tensor().long()
            pred_all = torch.Tensor()
        tq = tqdm.tqdm(total=len(val_dataloader) * batch_size)
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            output = F.softmax(output, dim=1)
            it_count += 1
            label_all = torch.cat((label_all, target), 0)
            pred_all = torch.cat((pred_all, output), 0)
#            print(pred_all)
            tq.update(batch_size)
        tq.close()
        output = pred_all
        target = label_all
        loss = criterion(output, target)
        loss_meter = loss.item()
        f1 = calc_f1(target, output)
    return loss_meter, f1, output


def train(train_dataloader, val_dataloader, batch_size):
    model = resnet18()
    model = model.to(device)
    # optimizer and loss
    lr = 0.00005
    optimizer = optim.Adam(model.parameters(), lr=lr)
#    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    # 模型保存文件夹
    model_save_dir = os.path.join('model', 'model' + str(fold_+1) + '.bin')
    best_f1 = -1
    start_epoch = 1
    tmp_epoch = 0

    # =========>开始训练<=========
    for epoch in range(start_epoch, max_epoch + 1):
        train_loss, train_f1 = train_epoch(model, optimizer, criterion,
                                           train_dataloader, epoch, lr, best_f1, batch_size)
        
        val_loss, val_f1, _ = val_epoch(model, criterion, val_dataloader, batch_size)

        tmp_epoch = tmp_epoch + 1
        if best_f1 < val_f1:
            tmp_epoch = 0
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_dir)
            print('save best f1:%.3f' % val_f1)
        if tmp_epoch > 2:
            print('early stop')
            break
    
    model.load_state_dict(best_state)
    return model


def test(val_dataloader, model, batch_size):
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            pred_all = torch.Tensor().cuda()
        else:
            pred_all = torch.Tensor()
        tq = tqdm.tqdm(total=len(val_dataloader) * batch_size)
        for inputs in val_dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            output = F.softmax(output, dim=1)
            pred_all = torch.cat((pred_all, output), 0)
            tq.update(batch_size)
        tq.close()
        output = pred_all
    return output.cpu().detach().numpy()
    

if __name__=='__main__':
    
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)
    trn_len = 7000
    test_len = 2000
    oof_lgb = np.zeros((trn_len, 3))
    prediction = np.zeros((test_len, 3))

    criterion = nn.CrossEntropyLoss()
    all_dataset = MyDataset(trn_path, trn_figurepath)
    test_dataset = MyDataset(test_path, test_figurepath, False)
    test_loader = DataLoader(test_dataset, batch_size=trn_batchsize, num_workers=4, shuffle=False)
    labels = np.array(get_label(trn_path))
    import torch.optim as optim

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(np.zeros(trn_len))):
        print("fold n°{}".format(fold_ + 1))        

        train_5 = Subset(all_dataset, trn_idx)
        valid_5 = Subset(all_dataset, val_idx)

        train_loader = DataLoader(train_5, batch_size=trn_batchsize, num_workers=4, shuffle=True)
        val_loader = DataLoader(valid_5, batch_size=trn_batchsize, num_workers=4, shuffle=False)
        model = train(train_loader, val_loader, trn_batchsize)
        
        _, _, output = val_epoch(model, criterion, val_loader, trn_batchsize)
        oof_lgb[val_idx] = output.cpu().detach().numpy()
        prediction = prediction + test(test_loader, model, trn_batchsize)
        
    oof_lgb_final = np.argmax(oof_lgb, axis=1)  
    print(f1_score(labels, oof_lgb_final, average='macro'))
    
    for i in range(3):
        tp = sum((labels == i) & (oof_lgb_final == i))
    #     fp = sum((y_train != i) & (oof_lgb_final == i))
        recall = tp / np.sum(labels == i)
        precision = tp / np.sum(oof_lgb_final == i)
        fscore = 2 * recall * precision / (recall + precision)
        print('%d° recall: %.3f, precision: %.3f, fscore: %.3f' % (i, recall, precision, fscore))
    
    pred_label = np.argmax(prediction, axis=1)
    label_dict = {0:'拖网', 1:'刺网', 2:'围网'}
    df_pred = pd.DataFrame()
    df_pred['filename'] = list(range(test_len))
    df_pred['label'] = pred_label
    df_pred['label'] = df_pred['label'].map(label_dict)
    df_pred.to_csv('sub.csv', index=None, header=False)

    

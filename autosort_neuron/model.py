import torch.nn.functional as nnf
import torch
import torch.nn as nn
import numpy as np


def load_day_length(path):
    day_length = np.load(path)

    day_id = []
    for i in range(day_length.shape[0]):
        if i == 0:
            day_id.append(day_length[i])
        else:
            append_i = day_id[i - 1] + day_length[i]
            day_id.append(append_i)

    day_id.insert(0, 0)
    return day_id


class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=(ch))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.way2 = nn.Sequential(
            nn.Linear(1000, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.way3 = nn.Sequential(
            nn.Linear(512, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(100, num_classes, bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return x


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 5, padding=2)
        self.bn = torch.nn.BatchNorm1d(1, eps=0.001, momentum=0.99)
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        features = self.conv1(features.reshape(features.shape[0], 1, features.shape[1]))
        features = torch.relu(self.bn(features)).reshape(
            features.shape[0], features.shape[2]
        )
        activation = self.encoder_hidden_layer(features)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        reconstructed = self.decoder_output_layer(activation)
        # reconstructed = torch.relu(reconstructed)
        return code, reconstructed



class AutoSort:
    def __init__(self,  ch_num, samplepoints, 
    loc_dim, device, set_shank_id,save_dir, pos_weight_noise=None,pos_weight_label=None):
        self.clsfier_noise = clssimp((ch_num+1)*samplepoints+loc_dim , 2).to(device)
        self.clsfier_label = clssimp((ch_num+1)*samplepoints+loc_dim , len(set_shank_id)).to(device)

        self.optimizer = torch.optim.Adam([
                                {'params': self.clsfier_noise.parameters()},
                                {'params': self.clsfier_label.parameters()},
                                ], lr=1e-4)


        self.criterion = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_noise)
        self.bceloss_label = nn.BCEWithLogitsLoss(pos_weight=pos_weight_label)
        

        self.save_model_path_1 = save_dir+ 'multitask_single_wave_noise_ae.pth'
        self.save_model_path_2 = save_dir+ 'multitask_single_wave_clsfier_noise_clsfier.pth'
        self.save_model_path_3 = save_dir+ 'multitask_single_wave_clsfier_label_clsfier.pth'

        self.set_shank_id = set_shank_id

    def save_model(self):
        torch.save(self.clsfier_noise.state_dict(), self.save_model_path_2)
        torch.save(self.clsfier_label.state_dict(), self.save_model_path_3)

    def load_model(self):
        # self.model.load_state_dict(torch.load(save_model_path_1))
        self.clsfier_noise.load_state_dict(torch.load(self.save_model_path_2))
        self.clsfier_label.load_state_dict(torch.load(self.save_model_path_3))

    def to_device(self, device):
        self.clsfier_noise.to(device)
        self.clsfier_label.to(device)
        self.bceloss.to(device)
        self.bceloss_label.to(device)

    def train(self):
        self.clsfier_noise.train()
        self.clsfier_label.train()
        
    def eval(self):
        self.clsfier_noise.eval()
        self.clsfier_label.eval()      

    def iter_model(self, batch_features, classify_labels, labels,
        single_waveform,pred_loc):
        self.optimizer.zero_grad()

        codes=batch_features
        codes = torch.cat((codes, single_waveform), axis=1)
        codes = torch.cat((codes, pred_loc), axis=1)

        cls_output = self.clsfier_noise(codes.float())


        test = labels[:,1]==1
        if sum(test)>1:
            cls_label_output = self.clsfier_label(codes.float()[test,:])
            train_loss3 = 1000*  self.bceloss_label(cls_label_output, classify_labels[test,:len(self.set_shank_id)])
        else:
            train_loss3 = torch.tensor(0)


        train_loss1 = 0 
        train_loss2 = 1000*  self.bceloss(cls_output, labels)


        train_loss = train_loss1 + train_loss2 + train_loss3
        train_loss.backward()
        self.optimizer.step()
        return train_loss1, train_loss2.item(), train_loss3.item(), test


    def iter_model_eval(self, batch_features, classify_labels, labels,
        single_waveform,pred_loc):

        codes=batch_features
        codes = torch.cat((codes, single_waveform), axis=1)
        codes = torch.cat((codes, pred_loc), axis=1)

        cls_output = self.clsfier_noise(codes.float())
        gt = torch.argmax(labels, axis=1)
        pred = torch.argmax(cls_output, axis=1)

        test = labels[:,1]==1
        if sum(test)>1:
            cls_label_output = self.clsfier_label(codes.float()[test,:])
            pred_class = torch.argmax(cls_label_output,axis=1)
            gt_label_class = torch.argmax(classify_labels[test, :len(self.set_shank_id)], axis=1)
            train_loss3 = 1000*  self.bceloss_label(cls_label_output, classify_labels[test,:len(self.set_shank_id)])
        else:
            train_loss3 = torch.tensor(0)
            gt_label_class=torch.tensor([])
            pred_class=torch.tensor([])

        train_loss1 = 0 
        train_loss2 = 1000*  self.bceloss(cls_output, labels)
        train_loss = train_loss1 + train_loss2 + train_loss3
        return train_loss1, train_loss2.item(), train_loss3.item(), gt, pred, gt_label_class, pred_class


    def iter_model_eval_umap(self, batch_features, classify_labels, labels,
        single_waveform,pred_loc):

        codes=batch_features
        codes = torch.cat((codes, single_waveform), axis=1)
        codes = torch.cat((codes, pred_loc), axis=1)

        cls_output = self.clsfier_noise(codes.float())
        codestest = self.clsfier_noise.intermediate_forward(codes.float())

        gt = torch.argmax(labels, axis=1)
        pred = torch.argmax(cls_output, axis=1)

        # test = cls_output[:,1]==1
        # if sum(test)>1:
        cls_label_output = self.clsfier_label(codes.float())
        codestest_label = self.clsfier_label.intermediate_forward(codes.float())

        pred_class = torch.argmax(cls_label_output,axis=1)
        gt_label_class = torch.argmax(classify_labels[:, :len(self.set_shank_id)], axis=1)

        # prob = nnf.softmax(cls_label_output, dim=1)

        return gt, pred, gt_label_class, pred_class, codestest, codestest_label

import os
import pandas as pd
import torch
from torch.utils.data import random_split
from waveform_loader import *
from tqdm import tqdm
from model import *
from sklearn.metrics import accuracy_score,f1_score
from pathlib import Path
from autosort_neuron.utils import seed_all

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print(available_gpus)




def run(args):

    seed_all(args.seed_all)

    args.ch_num = args.group.shape[0]
    args.loc_dim = args.sensor_positions_all.shape[1]
    print(args)


    set_day_id_str = args.day_id_str[args.set_time]
    data_path = args.cluster_path+"input/"+set_day_id_str

    # if args.load_model == '0':
    save_dir = args.cluster_path+"model_save/"+"train_day"+set_day_id_str+'_'+str(args.seed_all)+'/train_weight/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)


    
    set_channel_id = args.group
    sensor_positions = args.sensor_positions_all

    epochs = 20
    min_valid_loss = np.inf

    #### train AutoSort


    train_notpure_dataset = waveformLoader(data_path+'/train_data/',
                                            shank_channel=set_channel_id,
                                            sensor_positions=sensor_positions,
                                            )

        
    train, val = random_split(train_notpure_dataset, [int(len(train_notpure_dataset) * 0.8),
                                                        len(train_notpure_dataset) - int(
                                                            len(train_notpure_dataset) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True, )
    val_loader = torch.utils.data.DataLoader(val, batch_size=512, shuffle=False, )


    autosort_model = AutoSort(
                                args.ch_num, 
                                args.samplepoints, 
                                args.loc_dim, 
                                device, 
                                train_notpure_dataset.keep_id,
                                save_dir,
                                pos_weight_noise=train_notpure_dataset.pos_weight_noise,
                                pos_weight_label=train_notpure_dataset.pos_weight_label
                                )
    if os.path.exists(autosort_model.save_model_path_2):
        autosort_model.load_model()
        autosort_model.to_device(device)

    else:
        training_log = {'epoch': [],
                        'validation_acc_noise':[],
                        'validation_acc_label':[]}

        for epoch in range(epochs):
            training_log['epoch'].append(epoch + 1)
            print("epoch : {}/{}".format(epoch + 1, epochs))

            loss1 = 0
            loss2 = 0
            loss3 = 0
            autosort_model.train()
            autosort_model.bceloss.pos_weight = autosort_model.bceloss.pos_weight.to(device)
            autosort_model.bceloss_label.pos_weight = autosort_model.bceloss_label.pos_weight.to(device)
            for batch_features, classify_labels, labels, single_waveform,pred_loc in tqdm(train_loader):
                classify_labels = classify_labels.to(device)
                batch_features = batch_features.view(-1, args.samplepoints*args.ch_num).to(device)
                labels = labels.to(device)
                single_waveform  = single_waveform.to(device)
                pred_loc=torch.tensor(pred_loc).to(device)

                train_loss1, train_loss2, train_loss3, test = autosort_model.iter_model(batch_features, classify_labels, labels,single_waveform,pred_loc)

                loss1 += train_loss1
                loss2 += train_loss2
                if sum(test)>0:
                    loss3 += train_loss3

            loss1 = loss1 / len(train_loader)
            loss2 = loss2 / len(train_loader)
            loss3 = loss3 / len(train_loader)
            print("epoch : {}/{}, loss 1 = {:.6f}, loss 2 = {:.6f},, loss 3 = {:.6f}".format(epoch + 1, epochs, loss1,loss2,loss3))


            valid_loss1 = 0.0
            valid_loss2 = 0.0
            valid_loss3 = 0.0

            gt_all = []
            pred_all = []
            gt_class_all = []
            pred_class_all = []
            autosort_model.eval()
            for data, classify_labels, labels, single_waveform,pred_loc in tqdm(val_loader):
                classify_labels = classify_labels.to(device)
                data = data.view(-1, args.samplepoints * args.ch_num).to(device)
                labels = labels.to(device)
                single_waveform  = single_waveform.to(device)
                pred_loc=torch.tensor(pred_loc).to(device)

                valid_loss1, valid_loss2, valid_loss3, gt, pred, gt_label_class, pred_class = autosort_model.iter_model_eval(data, classify_labels, labels,single_waveform,pred_loc)

                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                pred_class_all.append(pred_class.detach().cpu().numpy())
                gt_class_all.append(gt_label_class.detach().cpu().numpy())

            gt_all = np.concatenate(gt_all, axis=0)
            pred_all = np.concatenate(pred_all, axis=0)
            gt_class_all = np.concatenate(gt_class_all, axis=0)
            pred_class_all = np.concatenate(pred_class_all, axis=0)

            valid_loss1 = valid_loss1 / len(val_loader)
            valid_loss2 = valid_loss2 / len(val_loader)
            valid_loss3 = valid_loss3 / len(val_loader)
            valid_loss = valid_loss1 +valid_loss2 + valid_loss3
            print("epoch : {}/{}, val loss 1 = {:.6f}, loss 2 = {:.6f},loss 3 = {:.6f}".format(epoch + 1, epochs, valid_loss1, valid_loss2, valid_loss3))


            training_log['validation_acc_noise'].append(accuracy_score(gt_all, pred_all))
            training_log['validation_acc_label'].append(f1_score(gt_class_all, pred_class_all,average='micro'))

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                autosort_model.save_model()

        pd.DataFrame(training_log).to_csv(save_dir+'training_log.csv')


    #### validate AutoSort
    if args.mode=='train':
        test_log = {'train_time':[], 'timepoint':[],'noise_acc':[],'label_acc':[],}
        for test_time in args.test_time:
            set_day_id_str = args.day_id_str[test_time]

            test_notpure_dataset = waveformLoader(data_path+'/test_data/',
                                            shank_channel = set_channel_id,
                                            sensor_positions=sensor_positions,
                                            Keep_id = train_notpure_dataset.keep_id,
                                            )


            test_notpure_loader = torch.utils.data.DataLoader(test_notpure_dataset,
            batch_size=512, shuffle=False,)

            gt_all = []
            pred_all = []
            gt_class_all = []
            pred_class_all = []
            autosort_model.eval()
            for data, classify_labels, labels, single_waveform,pred_loc in tqdm(test_notpure_loader):
                classify_labels = classify_labels.to(device)
                data = data.view(-1, args.samplepoints * args.ch_num).to(device)
                labels = labels.to(device)
                single_waveform  = single_waveform.to(device)
                pred_loc=torch.tensor(pred_loc).to(device)
                _,_,_, gt, pred, gt_label_class, pred_class = autosort_model.iter_model_eval(data, classify_labels, labels, single_waveform,pred_loc)
                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                pred_class_all.append(pred_class.detach().cpu().numpy())
                gt_class_all.append(gt_label_class.detach().cpu().numpy())


            gt_all = np.concatenate(gt_all, axis=0)
            pred_all = np.concatenate(pred_all, axis=0)
            gt_class_all = np.concatenate(gt_class_all, axis=0)
            pred_class_all = np.concatenate(pred_class_all, axis=0)

            test_log['train_time'].append(args.day_id_str[args.set_time])
            test_log['timepoint'].append(set_day_id_str)
            test_log['noise_acc'].append(accuracy_score(gt_all, pred_all))
            test_log['label_acc'].append(accuracy_score(gt_class_all, pred_class_all))

        pd.DataFrame(test_log).to_csv(save_dir+'test_log.csv')

    if args.mode=='test':
        save_dir_offline = args.cluster_path+"model_save/"+"train_day"+set_day_id_str+'_'+str(args.seed_all)+'/offline_result/'
        Path(save_dir_offline).mkdir(parents=True, exist_ok=True)
        for test_time in args.test_time:
            set_day_id_str = args.day_id_str[test_time]

            test_notpure_dataset = waveformLoader(data_path+'/test_data/',
                                            shank_channel = set_channel_id,
                                            sensor_positions=sensor_positions,
                                            Keep_id = train_notpure_dataset.keep_id,
                                            )
            
            test_notpure_loader = torch.utils.data.DataLoader(test_notpure_dataset,
                                                              batch_size=512, shuffle=False, )

            gt_all = []
            pred_all = []
            gt_class_all = []
            pred_class_all = []
            code_all_latent = []
            code_all_label = []
            autosort_model.eval()
            for data, classify_labels, labels, single_waveform, pred_loc in tqdm(test_notpure_loader):
                classify_labels = classify_labels.to(device)
                data = data.view(-1, args.samplepoints * args.ch_num).to(device)
                labels = labels.to(device)
                single_waveform = single_waveform.to(device)
                pred_loc = torch.tensor(pred_loc).to(device)

                gt, pred, gt_label_class, pred_class, codetest, codetest_label = autosort_model.iter_model_eval_umap(
                    data, classify_labels, labels, single_waveform, pred_loc)

                code_all_latent.append(codetest.detach().cpu().numpy())
                code_all_label.append(codetest_label.detach().cpu().numpy())

                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                pred_class_all.append(pred_class.detach().cpu().numpy())
                gt_class_all.append(gt_label_class.detach().cpu().numpy())

            gt_all = np.concatenate(gt_all, axis=0)
            pred_all = np.concatenate(pred_all, axis=0)
            gt_class_all = np.concatenate(gt_class_all, axis=0)
            pred_class_all = np.concatenate(pred_class_all, axis=0)
            code_all_latent = np.vstack(code_all_latent)
            code_all_label_final = []
            for i in code_all_label:
                if i.shape[0] > 0:
                    code_all_label_final.append(i)
            code_all_label_final = np.vstack(code_all_label_final)

            Path(save_dir_offline + set_day_id_str + '/').mkdir(parents=True, exist_ok=True)
            pd.DataFrame(gt_all).to_csv(save_dir_offline + set_day_id_str + '/' + 'gt_all.csv')
            pd.DataFrame(pred_all).to_csv(save_dir_offline + set_day_id_str + '/' + 'pred_all.csv')
            pd.DataFrame(gt_class_all).to_csv(save_dir_offline + set_day_id_str + '/' + 'gt_class_all.csv')
            pd.DataFrame(pred_class_all).to_csv(save_dir_offline + set_day_id_str + '/' + 'pred_class_all.csv')
            pd.DataFrame(code_all_latent).to_csv(save_dir_offline + set_day_id_str + '/' + 'code_all_latent.csv')
            pd.DataFrame(code_all_label_final).to_csv(save_dir_offline + set_day_id_str + '/' + 'code_all_label.csv')

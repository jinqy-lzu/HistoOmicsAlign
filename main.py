import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
import argparse
import os
import random
import tensorboard_logger as tb_logger
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from model import HistopGeneFusion, l1_reg_all
from dataset import geneHistopDataloader_surv, geneHistopDataloader_cls
from datetime import datetime
from optim import build_optimizer
from schedule import build_scheduler
from model_ema import ModelEMA
from misc import CheckpointManager, init_logger
from utils import CrossEntropySurvLoss,NLLSurvLoss,CoxSurvLoss
from sksurv.metrics import concordance_index_censored
import logging
import time
import math
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from args import parse_args
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count   
def draw_auc(y_score, y_true, path):
    for i,item in enumerate(y_score):
        if i == 0:
            score = item
        else:  
            score  = torch.cat((score, item), dim=0)  
    y_score = np.array(score.cpu().detach())
    
    for i,item in enumerate(y_true):
        if i == 0:
            score = item
        else:  
            score  = torch.cat((score, item), dim=0)  
    y_true = np.array(score.cpu().detach())

    n_classes = y_score.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        logger.info(f'Subtypes {i+1} (area = {roc_auc[i]:.4f})')
        plt.plot(fpr[i], tpr[i], label=f'Subtypes {i+1} (area = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    print('AUC-predict_depth:',roc_auc)
    plt.legend()           
    plt.xlabel(u'False Positive Rate')             
    plt.ylabel(u'True Positive Rate')                 
    plt.savefig('{}/roc-auc.pdf'.format(path),format='pdf') 
    plt.close()

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # for k in topk:
        correct_k = correct[:topk].float().sum()
        return correct_k.mul_(1.0 / batch_size)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
      
def unsqueeze_label(label, class_num,device):  
    batch_size = label.shape[0]
    label_tensor = torch.zeros(batch_size, class_num)
    for i in range(batch_size):
        index = label[i].item()
        label_tensor[int(i), int(index)] = 1
    return label_tensor.cuda(device)
           
def train_surv(model, model_ema, trainDataloader, criterion, optimizer,device,scheduler,args,epoch):
    train_loss = AverageMeter()
    model.train()
    start = time.time() 
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_sample_haz = []
    all_sample_id = []
    reg_fn = l1_reg_all
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, event_time,c, sample_id = data 
        gene = gene.to(torch.float)
        event_time = event_time.to(torch.float)
        c = c.to(torch.float)
        gene = gene.cuda()
        batch_y = batch_y.cuda()
        img = img.cuda()
        event_time = event_time.cuda()
        c = c.cuda()
        label = batch_y
        optimizer.zero_grad()
        hazards, S, Y_hat, _  = model(gene, img)
        loss = criterion(hazards=hazards, S=S, Y=label, c=c)
        risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
        censorships = c.detach().cpu().numpy()
        event_times = event_time.detach().cpu().numpy()
        all_risk_scores.append(risk_scores)
        all_censorships.append(censorships)
        all_event_times.append(event_times)
        all_sample_haz.append(hazards.detach().cpu().numpy())
        all_sample_id.append(sample_id)

        loss_reg = reg_fn(model) * args.lambda_reg
        loss = loss + loss_reg

        loss.backward()
        optimizer.step()
        train_loss.update(loss,label.size(0))
        end = time.time()
        if args.sched != "step":
            scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_sample_haz = np.concatenate(all_sample_haz)
    all_sample_id = np.concatenate(all_sample_id)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    logger.info('Train: {} | ' 'C-index: {:.4f} | ' 'Loss: {:.8f} | '
                'LR: {:.3e} | ' 'Time:({:.2f}s) '
                        .format(
                            epoch,
                            all_c_index,
                            train_loss.avg,
                            optimizer.param_groups[0]['lr'],
                            end-start,
                            ))        
    return {'top1': all_c_index,"train_loss":train_loss.avg},all_censorships,all_event_times,all_risk_scores,all_sample_haz,all_sample_id

def test_surv(model, testDataloader, criterion, device,optimizer, epoch, logsuffix, args):
    model.eval()
    test_loss = AverageMeter()
    err = AverageMeter()
    all_labels = []
    sur_OS = []
    OS_time = []
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_hazards = []
    all_sample_id = []
    with torch.no_grad():
        for id, data in enumerate(testDataloader):
            batch_x,batch_y,batch_img, event_time, c, sample_id  = data
            sur_OS.append(c)
            OS_time.append(event_time)

            batch_x = batch_x.to(torch.float)
            event_time = event_time.to(torch.float)
            c = c.to(torch.float)
            batch_x = batch_x.cuda()
            label = batch_y.cuda()
            batch_img = batch_img.cuda()
            event_time = event_time.cuda()
            c = c.cuda()

            hazards, S, Y_hat, _ = model(batch_x, batch_img)
            
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
            censorships = c.detach().cpu().numpy()
            event_times = event_time.detach().cpu().numpy()

            all_risk_scores.append(risk_scores)
            all_censorships.append(censorships)
            all_event_times.append(event_times)
            all_hazards.append(hazards.detach().cpu().numpy())
            all_sample_id.append(sample_id)

            test_loss.update(loss,label.size(0))
            all_labels.extend(batch_y.tolist())

            err.update(calculate_error(Y_hat, label), label.size(0))
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_hazards = np.concatenate(all_hazards)
    all_sample_id = np.concatenate(all_sample_id)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    all_labels_tensor = torch.tensor(all_labels)
    unique_labels, counts = torch.unique(all_labels_tensor, return_counts=True)
    print(f"test data Unique labels: {unique_labels}")
    print(f"test data Counts: {counts}")
    logger.info('{}: {} | ' 'C-index: {:.4f} | ' 'Loss: {:.8f} |  ERR:{} |'
                'LR: {:.3e} '.format(
                            logsuffix,
                            epoch,
                            all_c_index,
                            test_loss.avg,
                            err.avg,
                            optimizer.param_groups[0]['lr']
                            ))
    return {"test_loss": test_loss.avg,"top1": all_c_index},all_censorships,all_event_times,all_risk_scores,all_hazards,all_sample_id

def train_cls(model, model_ema, trainDataloader, criterion, optimizer,device,scheduler,args,epoch):
    top1 = AverageMeter()
    train_loss = AverageMeter()
    model.train()
    start = time.time() 
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img = data
        gene = gene.to(torch.float)
        batch_y = batch_y.to(torch.float)
        gene = gene.cuda(device)
        batch_y = batch_y.cuda(device)
        img = img.cuda(device)
        label = unsqueeze_label(batch_y, args.classes, device)
        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = None
        outputs = model(gene, img)
        loss = criterion(outputs, label)
        loss_reg = reg_fn(model) * args.lambda_reg
        loss = loss + loss_reg

        acc1 = accuracy(outputs, batch_y, topk=(1))
        loss.backward()
        optimizer.step()
        top1.update(acc1, label.size(0))
        train_loss.update(loss,label.size(0))
        end = time.time()
        if args.sched != "step":
            scheduler.step()
        if model_ema is not None:
            model_ema.update(model)
    logger.info('Train: {} | ' 'Acc: {:.4f} | ' 'Loss: {:.3f} | '
                'LR: {:.3e} | ' 'Time:({:.2f}s) '
                        .format(
                            epoch,
                            top1.avg,
                            train_loss.avg,
                            optimizer.param_groups[0]['lr'],
                            end-start,
                            ))        
    return {'top1': top1.avg ,"train_loss":train_loss.avg}

def test_cls(model, testDataloader, criterion, device, optimizer, epoch, logsuffix, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    top1 = AverageMeter()
    test_loss = AverageMeter()
    label_list = []
    predict_list = []
    all_labels = []
    with torch.no_grad():
        for id, data in enumerate(testDataloader):
            batch_x,batch_y,batch_img = data
            label_list.append(batch_y)
            batch_x = batch_x.to(torch.float)
            batch_y = batch_y.to(torch.float)
            batch_x = batch_x.cuda(device)
            batch_y = batch_y.cuda(device)
            batch_img = batch_img.cuda(device)
            outputs = model(batch_x, batch_img)
            predict_list.append(outputs)
            label = unsqueeze_label(batch_y, args.classes, device)
            loss = criterion(outputs, label)

            acc1= accuracy(outputs, batch_y, topk=1)
            top1.update(acc1, batch_y.size(0))
            test_loss.update(loss,label.size(0))
            all_labels.extend(batch_y.tolist())

    all_labels_tensor = torch.tensor(all_labels)
    unique_labels, counts = torch.unique(all_labels_tensor, return_counts=True)
    print(f"test data Unique labels: {unique_labels}")
    print(f"test data Counts: {counts}")
    logger.info('{}: {} | ' 'Acc: {:.4f} | ' 'Loss: {:.3f} | '
                'LR: {:.3e} '.format(
                            logsuffix,
                            epoch,
                            top1.avg,
                            test_loss.avg,
                            optimizer.param_groups[0]['lr']
                            ))
    return {"test_loss": test_loss.avg,"top1": top1.avg}, predict_list, label_list

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

def main():
    
    args, args_text = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = int(args.gpu_id)
    lucky_number = 3407 #114514 3407
    seed_torch(lucky_number)
    
    init_logger(args)
    # save args
    logger.info(args)

    time_str = f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    t_logger = tb_logger.Logger(logdir='{}/{}/'.format(args.best_model_path, time_str, flush_secs=2))
    if args.task_type == 'survival':
        trainDataloader, testDataloader,gene_size, img_feat_size = geneHistopDataloader_surv(geneFile=args.gene_dir, selectData=None, imgDir=args.img_dir,
                                           batchSize=args.batch_size, train_ratio=args.train_ration)
    else:
        trainDataloader, testDataloader,gene_size, img_feat_size = geneHistopDataloader_cls(geneFile=args.gene_dir, selectData=None, imgDir=args.img_dir,
                                           batchSize=args.batch_size, train_ratio=args.train_ration)

    l = gene_size[0]
    ch, emd = img_feat_size

    model = HistopGeneFusion(gene_nums=l,img_feat_ch=ch, img_feat_emb=emd,classes=args.classes,mode = args.task_type)
    logger.info(model)

    model.cuda()
    optimizer = build_optimizer(args.opt,
                                model,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.best_model_path
                                     )
    loss_fn = None
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    criterion = loss_fn
    cudnn.benchmark = True

    
    if args.sched != "step":
        steps_per_epoch = len(trainDataloader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        decay_steps = args.decay_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        scheduler = build_scheduler(args.sched,
                                    optimizer,
                                    warmup_steps,
                                    args.warmup_lr,
                                    decay_steps,
                                    args.decay_rate,
                                    total_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    decay_by_epoch=args.decay_by_epoch,
                                    min_lr=args.min_lr)
    else:
        scheduler=None
    
    if args.task_type == 'survival':
        best_acc = 0
        for epoch in range(args.epochs):
            if args.sched == "step":          
                adjust_learning_rate(optimizer, epoch, args)
            metrics,train_sur_os,train_sur_time,train_sur_risk,train_surv_hazards,train_sample_id = train_surv(model, model_ema, trainDataloader, criterion, optimizer, device, scheduler, args, epoch)
            t_logger.log_value('train C-index', metrics['top1'], epoch)
            t_logger.log_value('train_loss', metrics['train_loss'], epoch)
            
            metrics, sur_os,sur_time,sur_risk,surv_hazards,sample_id = test_surv(model, testDataloader, criterion, device,optimizer, epoch, "Test:", args)
            if model_ema is not None:
            test(model_ema.module, testDataloader, criterion, device, optimizer, epoch,"EMA:",args)
            t_logger.log_value('test C-index', metrics['top1'], epoch)
            t_logger.log_value('test_loss', metrics['test_loss'], epoch)
            # checkpoint
            ckpts = ckpt_manager.update(epoch, metrics)
            if best_acc <= metrics['top1']:
                best_acc = metrics['top1']
                col_name1 = []
                survival_df = pandas.DataFrame({"ID": sample_id,"OS":sur_os,"sur_time":sur_time,"risk":sur_risk})
                for i in range(0, surv_hazards.shape[1]):
                    col_name1.append(f"{i}_hazard")
                surv_hazards = pandas.DataFrame(surv_hazards, columns=col_name1)
                survival_df = pandas.concat([survival_df, surv_hazards], axis=1)
                file = f'test_{time_str}_survival_risk.csv'
                survival_df.to_csv(os.path.join(args.sur_ret_dir, file))

                col_name2 = []
                survival_df_train = pandas.DataFrame({"ID":train_sample_id,"OS":train_sur_os,"sur_time":train_sur_time,"risk":train_sur_risk})
                for i in range(0, train_surv_hazards.shape[1]):
                    col_name2.append(f"{i}_hazard")
                train_surv_hazards = pandas.DataFrame(train_surv_hazards, columns=col_name2)
                survival_df = pandas.concat([survival_df, train_surv_hazards], axis=1)
                file = f'train_{time_str}_survival_risk.csv'
                survival_df_train.to_csv(os.path.join(args.sur_ret_dir, file))
                logger.info("save predict risk!!!!!")  
    else:
        best_acc = 0
        score_auc = []
        label_auc = []
        for epoch in range(args.epochs):
            if args.sched == "step":          
                adjust_learning_rate(optimizer, epoch, args)
            metrics = train_cls(model, model_ema, trainDataloader, criterion, optimizer, device, scheduler, args, epoch)
            t_logger.log_value('train_acc', metrics['top1'], epoch)
            t_logger.log_value('train_loss', metrics['train_loss'], epoch)
            
            metrics, score_list, label_list = test_cls(model, testDataloader, criterion, device,optimizer, epoch, "Test:", args)
            if model_ema is not None:
            test(model_ema.module, testDataloader, criterion, device, optimizer, epoch,"EMA:",args)
            t_logger.log_value('test_acc', metrics['top1'], epoch)
            t_logger.log_value('test_loss', metrics['test_loss'], epoch)
            # checkpoint
            ckpts = ckpt_manager.update(epoch, metrics)
            if best_acc <= metrics['top1']:
                best_acc = metrics['top1']
                score_auc = score_list
                label_auc = label_list
                draw_auc(score_auc, label_auc, args.best_model_path)

if __name__ == '__main__':
    main()
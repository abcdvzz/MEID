import os
import copy
import torch
import shutil
import time
import warnings
import numpy as np
import random
from ops import Augment
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from opts import parser
from ops.mapmeter import mAPMeter, LTMeter

from ops.apmeter import APMeter

from ops.utils import AverageMeter, accuracy

from ops import losses
from tools import utils

from dataset import dutils
from models import models
from ops.feature_loader import BasicDataset, ResamplingDataset_Mask, LTPlusDataset, LTPlusDataset_diff, ResamplingDataset_Mask_diff, LTPlusDataset_diff_charades, ResamplingDataset_Mask_diff_charades

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            if dist.get_rank() == 0:
                print('creating folder ' + folder)
                os.mkdir(folder)


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def load_data(num_class, input_dir, dataset='VideoLT'):
    train_list = open(args.train_list, 'r').readlines()
    val_list = open(args.val_list, 'r').readlines()
    
    if args.resample == 'None':
        if dataset == 'Charades' or dataset == 'CharadesEgo':
            train_dataset = LTPlusDataset_diff_charades(train_list, input_dir, args.train_num_frames, cls_num=args.num_class, train_mode=True)
        else:
            train_dataset = LTPlusDataset_diff(train_list, input_dir, args.train_num_frames, cls_num=args.num_class, train_mode=True)
    else:
        if dataset == 'Charades' or dataset == 'CharadesEgo':
            train_dataset = ResamplingDataset_Mask_diff_charades(train_list, input_dir, args.train_num_frames, \
            rstype=args.resample, cls_num=args.num_class, train_mode=True)
        else:
            train_dataset = ResamplingDataset_Mask_diff(train_list, input_dir, args.train_num_frames, \
            rstype=args.resample, cls_num=args.num_class, train_mode=True)
    if dataset == 'Charades' or dataset == 'CharadesEgo':
        val_dataset = LTPlusDataset_diff_charades(val_list, input_dir, args.val_num_frames, \
            cls_num=args.num_class, train_mode=False)
    else:
        val_dataset = LTPlusDataset_diff(val_list, input_dir, args.val_num_frames, \
            cls_num=args.num_class, train_mode=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_dataloader = DataLoaderX(train_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    val_dataloader = DataLoaderX(val_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=args.workers, pin_memory=False, sampler=val_sampler)


    return train_dataloader, val_dataloader


def main():

    global args, best_mAP, criterion, optimizer, tf_writer, log_training
    
    best_mAP = 0
    ap = 0
    ap1 = 0
    args = parser.parse_args()
    start_epoch = args.start_epoch 
    num_class = args.num_class

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    # torch.autograd.set_detect_anomaly(True)
    
    if args.resample != 'None':
        args.reduce = "none"

    if dist.get_rank() == 0:
        print ("########################################################################\n")
        print ("Feature name: {} \nNumber of class: {} \nTrain frames: {} \nVal frames: {}\nReduction: {}".\
                format(args.feature_name, args.num_class, args.train_num_frames, args.val_num_frames, args.reduce))
        print ("Applied long-tailed strategies: \n")
        print ("\tAugmentation: {} \t Re-weighting: {} \t Re-sampling: {} \n". \
                format("FrameStack", args.loss_func, args.resample))
        print ("######################################################################## \n") 
    check_rootfolders()
    setup_seed(args.seed)
    
    input_dir =  dutils.get_feature_path(args.feature_name)
    feature_dim = dutils.get_feature_dim(args.feature_name)
    args.lc_list, args.train_list, args.val_list = dutils.get_lt_plus_path(dataset=args.feature_name)

    train_loader, val_loader = load_data(num_class, input_dir,dataset=args.feature_name)

    criterion = utils.find_class_by_name(args.loss_func, [losses])(args, logits=True, reduce="mean")

    criterion = criterion.to(local_rank)
    
    indices = utils.get_indices(args.lc_list, head=args.head, tail=args.tail,dataset=args.feature_name)
    
    model = utils.find_class_by_name(args.model_name, [models])(feature_dim, num_class) 
    # model = model.cuda()
    model = model.to(local_rank)

    if args.resume != "": 
        if dist.get_rank() == 0:
            print ("=> Loading checkpoint {}".format(args.resume))
        if dist.get_rank() == 0:
            ckpt = torch.load(args.resume)
            best_mAP = ckpt['best_mAP']
            start_epoch = ckpt['epoch'] + 1
            acc1 = ckpt['Acc@1']
            acc5 = ckpt['Acc@5']
            sd = ckpt['state_dict']
        
        if dist.get_rank() == 0:
            print ("Loaded checkpoint {} epoch {}: best_mAP {} | Acc@1 {} | Acc@5 {}". \
                format(args.resume, start_epoch, best_mAP, acc1, acc5))
        
            # model.load_state_dict(sd)
            model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()})

    if args.pretrain != "": 
        if dist.get_rank() == 0:
            print ("=> Loading pretrained model {}".format(args.pretrain))
        if dist.get_rank() == 0:
            ckpt = torch.load(args.pretrain)
            # model.load_state_dict(ckpt['state_dict'])
            model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['state_dict'].items()}, strict=False)

    # model=torch.nn.DataParallel(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,find_unused_parameters=True)

    if dist.get_rank() == 0:
        print ("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            if dist.get_rank() == 0:
                print ('\t', name)
    
    optimizer = torch.optim.Adam(params_to_update, lr=args.lr)
    
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'),'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        if dist.get_rank() == 0:
            print ("Training for Epoch {}".format(epoch))
        if epoch > args.warm_epoch:
            if dist.get_rank() == 0:
                print("Start dynamic training for epoch {}......\n".format(epoch))
        
        train_loader.sampler.set_epoch(epoch)
        
        if args.resample == 'None':
            ap, ap1 = train(train_loader, model, epoch, log_training, indices, ap, ap1)
        else:
            ap = rs_train(train_loader, model, epoch, log_training, indices, ap)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            acc1, acc5, mAP = validate(val_loader, model, epoch, log_training, indices)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            tf_writer.add_scalar('best_mAP/test_best', best_mAP, epoch)
            
            if dist.get_rank() == 0:
                print ('Test Epoch {}: Acc@1: {} | Acc@5: {} | mAP: {} | best_mAP: {}'.format(epoch, acc1, acc5, mAP, best_mAP))

                save_checkpoint({
                'epoch': epoch + 1,
                'feature': args.feature_name,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
                'Acc@1': acc1,
                'Acc@5': acc5},
                is_best)


def train(loader, model, epoch, log, indices, ap=0, ap1=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()
    mAP1 = mAPMeter()
    model.train()
    
    end = time.time()
    if dist.get_rank() == 0:
        print ("Ap for epoch {}: {}".format(epoch, ap))
        print ("Ap1 for epoch {}: {}".format(epoch, ap1))

    for i, (vid, feature, feature_diff, target) in enumerate(loader):

        feature = feature.to(args.local_rank)
        target = target.to(args.local_rank)
        feature_diff = feature_diff.cuda()

        if epoch <= args.warm_epoch:

            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(feature, feature_diff, phase='train')

            loss = criterion(output1, target)

        else:
            
            if args.ratio > 0:
                batch_size = feature.size(0)
                split_samples = int(batch_size * args.ratio)
                mixed_input, mixed_target = Augment.FrameStack(feature[:split_samples], target[:split_samples], args.clip_length, ap)
                mixed_input = torch.cat((mixed_input, feature[split_samples:]),dim=0)
                mixed_target = torch.cat((mixed_target, target[split_samples:]),dim=0)
                mixed_input_diff, _ = Augment.FrameStack(feature_diff[:split_samples], target[:split_samples], args.clip_length, ap)
                mixed_input_diff = torch.cat((mixed_input_diff, feature_diff[split_samples:]),dim=0)
            else:
                mixed_input, mixed_target = Augment.FrameStack(feature, target, args.clip_length, ap)

            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(mixed_input,mixed_input_diff, phase='train')
            loss = criterion(output1, mixed_target)

        losses.update(loss.item(), output.size(0))

        with torch.no_grad():

            all_output = (output + output1) /2

            prec1, prec5 = accuracy(all_output.data, target, topk=(1, 5))
            top1.update(prec1, output.size(0))
            top5.update(prec5, output.size(0))

            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(feature, feature_diff, phase='train')

            all_prediction = (prediction + prediction1) /2

            mAP.add(all_prediction, target)
            mAP1.add(prediction, target)

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                      .format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            if dist.get_rank() == 0:
                print(output)
            
            log.write(output)
            log.flush()
    tf_writer.add_scalar('loss/train_epoch', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    train_ap = mAP.value()
    train_ap1 = mAP1.value()
    if dist.get_rank() == 0:
        print ("mAP = ", mAP.avg())
        print ("mAP1 = ", mAP1.avg()) 
    return train_ap, train_ap1


def validate(loader, model, epoch, log, indices):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()
    
    LTmAP =LTMeter(indices)
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (vid, feature, feature_diff, target) in enumerate(loader):
            feature = feature.cuda()
            target = target.float().cuda()

            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(feature, feature_diff, phase='val')
            loss = criterion(output1, target)

            all_output = 0.5*output + 0.5*output1

            all_prediction = 0.5*prediction + 0.5*prediction1

            prec1, prec5 = accuracy(all_output.data, target, topk=(1, 5))

            losses.update(loss.item(), feature.size(0))

            top1.update(prec1, feature.size(0))
            top5.update(prec5, feature.size(0))

            mAP.add(all_prediction, target)
            LTmAP.add(all_prediction, target)

            batch_time.update(time.time() - end)
            end = time.time()
           
        head_map = LTmAP.value()["head"]
        medium_map = LTmAP.value()["medium"]
        tail_map = LTmAP.value()["tail"]

    output = ('Testing Results: Prec@1 {top1.avg:.5f} | Prec@5 {top5.avg:.5f} | Loss {loss.avg:.5f} | Loss1 {loss1.avg:.5f} | Loss2 {loss2.avg:.5f} | Loss3 {loss3.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))

    if dist.get_rank() == 0:
        print(output)
    map_output = ("Overall mAP = {:.3f} | Head = {:.5f} | Medium = {:.5f} | Tail = {:.5f}".format(mAP.avg(), head_map, medium_map, tail_map))
    if dist.get_rank() == 0:
        print (map_output)

    if log is not None:
        log.write(output + '  mAP {}\n'.format(mAP.avg()))
        log.write(map_output+'\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
        tf_writer.add_scalar('mAP/test', mAP.avg(), epoch)
    return top1.avg, top5.avg, mAP.avg()    


def rs_train(loader, model, epoch, log, indices, ap=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()
    
    model.train()
    end = time.time()
    
    if args.loss_func == 'LDAM':
        criterion.reset_epoch(epoch)

    if dist.get_rank() == 0:
        print ("Ap for epoch {}: {}".format(epoch, ap))
    for i, (vid, feature, feature_diff, target, mask) in enumerate(loader):
        # print('loading...')
        feature = feature.cuda()
        feature_diff = feature_diff.cuda()
        target = target.float().cuda(non_blocking=True)
        mask = mask.float().cuda()

        if epoch <= args.warm_epoch:
            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(feature, feature_diff, phase='train')
            loss = criterion(output1, target)

        else:
            if args.ratio > 0:
                batch_size = feature.size(0)
                split_samples = int(batch_size * args.ratio)
                mixed_input, mixed_target = Augment.FrameStack(feature[:split_samples], target[:split_samples], args.clip_length, ap)
                mixed_input = torch.cat((mixed_input, feature[split_samples:]),dim=0)
                mixed_target = torch.cat((mixed_target, target[split_samples:]),dim=0)
                mixed_input_diff, _ = Augment.FrameStack(feature_diff[:split_samples], target[:split_samples], args.clip_length, ap)
                mixed_input_diff = torch.cat((mixed_input_diff, feature_diff[split_samples:]),dim=0)
            else:
                mixed_input, mixed_target = Augment.FrameStack(feature, target, args.clip_length, ap)
            
            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(mixed_input, mixed_input_diff, phase='train')
            loss = criterion(output1, mixed_target)

        loss = loss * mask
        loss = torch.mean(torch.sum(loss, 1))
        losses.update(loss.item(), output.size(0))

        with torch.no_grad():

            all_output = 0.5*output + 0.5*output1

            prec1, prec5 = accuracy(all_output.data, target, topk=(1, 5))
            top1.update(prec1, output.size(0))
            top5.update(prec5, output.size(0))

            prediction, output, z, prediction1, output1, graph_output1, graph_output2 = model(feature, feature_diff, phase='train')

            all_prediction = 0.5*prediction + 0.5*prediction1

            mAP.add(all_prediction, target)

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                      .format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            if dist.get_rank() == 0:
                print(output)
            
            log.write(output)
            log.flush()
    tf_writer.add_scalar('loss/train_epoch', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    train_ap = mAP.value()
    if dist.get_rank() == 0:
        print ("mAP = ", mAP.avg()) 
    return train_ap


if __name__=='__main__':
    with torch.autograd.set_detect_anomaly(True):
        main()

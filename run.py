import random
import torch
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.nn.parallel
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import os
import re
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import argparse
import torch.distributed as dist
import time
import online_augmentation
import logging
from process_data.Load_data import DATA_process



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cross_entropy(logits, target):
    p = F.softmax(logits, dim=1)
    log_p = -torch.log(p)
    loss = target*log_p
    # print(target,p,log_p,loss)
    batch_num = logits.shape[0]
    return loss.sum()/batch_num



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def tensorboard_settings(args):
    if 'raw' in args.mode:
        if args.data_path:
            # raw_aug
            log_dir = os.path.join(args.output_dir, 'Raw_Aug_{}_{}_{}_{}_{}'.format(args.data_path.split(
                '/')[-1], args.seed, args.augweight, args.batch_size, args.aug_batch_size))
            if os.path.exists(log_dir):
                raise IOError(
                    'This tensorboard file {} already exists! Please do not train the same data repeatedly, if you want to train this dataset, delete corresponding tensorboard file first! '.format(log_dir))
            writer = SummaryWriter(log_dir=log_dir)
        else:
            # raw
            if args.random_mix:
                log_dir = os.path.join(args.output_dir, 'Raw_random_mixup_{}_{}_{}'.format(
                    args.random_mix, args.alpha, args.seed))
                if os.path.exists(log_dir):
                    raise IOError(
                    'This tensorboard file {} already exists! Please do not train the same data repeatedly, if you want to train this dataset, delete corresponding tensorboard file first! '.format(log_dir))
                writer = SummaryWriter(log_dir=log_dir)
            else:
                log_dir = os.path.join(
                    args.output_dir, 'Raw_{}'.format(args.seed))
                if os.path.exists(log_dir):
                    raise IOError(
                    'This tensorboard file {} already exists! Please do not train the same data repeatedly, if you want to train this dataset, delete corresponding tensorboard file first! '.format(log_dir))
                writer = SummaryWriter(log_dir=log_dir)
    elif args.mode == 'aug':
        # aug
        log_dir = os.path.join(args.output_dir, 'Aug_{}_{}_{}_{}_{}'.format(args.data_path.split(
            '/')[-1], args.seed, args.augweight, args.batch_size, args.aug_batch_size))
        if os.path.exists(log_dir):
                raise IOError(
                    'This tensorboard file {} already exists! Please do not train the same data repeatedly, if you want to train this dataset, delete corresponding tensorboard file first! '.format(log_dir))
        writer = SummaryWriter(log_dir=log_dir)
    return writer


def logging_settings(args):
    logger = logging.getLogger('result')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    if not os.path.exists(os.path.join('DATA', args.data.upper(), 'logs')):
        os.makedirs(os.path.join(
            'DATA', args.data.upper(), 'logs'))
    if args.low_resource_dir:
        log_path = os.path.join('DATA', args.data.upper(),'logs', 'lowresourcebest_result.log')
    else:
        log_path = os.path.join('DATA', args.data.upper(),'logs', 'best_result.log')
    
    fh = logging.FileHandler(log_path, mode='a+', encoding='utf-8')
    ft=logging.Filter(name='result.a')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    fh.addFilter(ft)
    logger.addHandler(fh)
    result_logger=logging.getLogger('result.a')
    return result_logger
def loading_model(args,label_num):
    t1 = time.time()
    if args.local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1  # the number of gpu on each proc
    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.cuda.device_count()
    else:
        args.world_size = 1
    print('*'*40, '\nSettings:{}'.format(args))
    print('*'*40)
    print('='*20, 'Loading models', '='*20)
    model = BertForSequenceClassification.from_pretrained(
        args.model, num_labels=label_num)
    model.to(device)
    t2 = time.time()
    print(
        '='*20, 'Loading models complete!, cost {:.2f}s'.format(t2-t1), '='*20)
    # model parrallel
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)
    if args.load_model_path is not None:
        print("="*20, "Load model from %s", args.load_model_path,)
        model.load_state_dict(torch.load(args.load_model_path))
    return model

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument(
        '--mode', type=str, choices=['raw', 'aug', 'raw_aug', 'visualize'], required=True)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num_proc', type=int, default=8,
                        help='multi process number used in dataloader process')

    # training settings
    parser.add_argument('--output_dir', type=str, help="tensorboard fileoutput directory")
    parser.add_argument('--epoch', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--seed', default=42, type=int, help='seed ')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='train examples in each batch')
    parser.add_argument('--val_steps', default=100, type=int,
                        help='evaluate on dev datasets every steps')
    parser.add_argument('--max_length', default=128,
                        type=int, help='encode max length')
    parser.add_argument('--label_name', type=str, default='label')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--low_resource_dir', type=str,
                        help='Low resource data dir')

    # train on augmentation dataset parameters
    parser.add_argument('--aug_batch_size', default=128,
                        type=int, help='train examples in each batch')
    parser.add_argument('--augweight', default=0.2, type=float)
    parser.add_argument('--data_path', type=str, help="augmentation file path")
    parser.add_argument('--min_train_token', type=int, default=0,
                        help="minimum token num restriction for train dataset")
    parser.add_argument('--max_train_token', type=int, default=0,
                        help="maximum token num restriction for train dataset")
    parser.add_argument('--mix', action='store_false', help='train on 01mixup')

    # random mixup
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="online augmentation alpha")
    parser.add_argument('--onlyaug', action='store_true',
                        help="train only on online aug batch")
    parser.add_argument('--difflen', action='store_true',
                        help="train only on online aug batch")
    parser.add_argument('--random_mix', type=str, help="random mixup ")

    # visualize dataset

    args = parser.parse_args()
    if args.data == 'trec':
        try:
            assert args.label_name in ['label-fine', 'label-coarse']
        except AssertionError:
            raise(AssertionError(
                "If you want to train on trec dataset with augmentation, you have to name the label of split"))
        if not args.output_dir:
            args.output_dir = os.path.join(
                'DATA', args.data.upper(), 'runs', args.label_name, args.mode)
    if args.mode == 'raw':
        args.batch_size = 128
    if 'aug' in args.mode:
        assert args.data_path
        if args.mode == 'aug':
            args.seed = 42
    if not args.output_dir:
        args.output_dir = os.path.join(
            'DATA', args.data.upper(), 'runs', args.mode)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.data in ['rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
        args.task = 'pair'
    else:
        args.task = 'single'

    return args


def train(args):
    # ========================================
    #         Tensorboard &Logging
    # ========================================
    writer = tensorboard_settings(args)
    result_logger = logging_settings(args)
    data_process = DATA_process(args)
    # ========================================
    #             Loading datasets
    # ========================================
    print('='*20, 'Start processing dataset', '='*20)
    t1 = time.time()

    val_dataloader = data_process.validation_data()
    
    if args.mode != 'aug':
        train_dataloader, label_num = data_process.train_data(count_label=True)
        # print('Label_num',label_num)
    if args.data_path:
        print('='*20, 'Train Augmentation dataset path: {}'.format(args.data_path), '='*20)
        aug_dataloader = data_process.augmentation_data()
        if args.mode == 'aug':
            train_dataloader = aug_dataloader
        else:
            aug_dataloader = cycle(aug_dataloader)

    t2 = time.time()
    print('='*20, 'Dataset process done! cost {:.2f}s'.format(t2-t1), '='*20)
    
    # ========================================
    #                   Model
    # ========================================
    model=loading_model(args,label_num)
    # ========================================
    #           Optimizer Settings
    # ========================================
    optimizer = AdamW(model.parameters(), lr=args.lr)
    all_steps = args.epoch*len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=all_steps)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # ========================================
    #               Train
    # ========================================
    print('='*20, 'Start training', '='*20)
    best_acc = 0
    args.val_steps = min(len(train_dataloader), args.val_steps)
    
    for epoch in range(args.epoch):
        bar = tqdm(enumerate(train_dataloader), total=len(
            train_dataloader)//args.world_size)
        fail = 0
        loss = 0
        for step, batch in bar:
            model.zero_grad()
         
            # ----------------------------------------------
            #               Train_dataloader
            # ----------------------------------------------
            if args.random_mix:
                try:
                    
                    input_ids, target_a = batch['input_ids'], batch['labels']
                    lam = np.random.choice([0, 0.1, 0.2, 0.3])
                    exchanged_ids, new_index = online_augmentation.random_mixup(
                        args, input_ids, target_a, lam)
                    target_b = target_a[new_index]
                    outputs = model(exchanged_ids.to(args.device), token_type_ids=None, attention_mask=(
                        exchanged_ids > 0).to(args.device))
                    logits = outputs.logits
                    loss = criterion(logits.to(args.device), target_a.to(
                        args.device))*(1-lam)+criterion(logits.to(args.device), target_b.to(args.device))*lam
                    
                    
                except Exception as e:
                    fail += 1
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
            elif args.model == 'aug':
                # train only on augmentation dataset
                batch = {k: v.to(args.device) for k, v in batch.items()}
                if args.mix:
                    # train on 01 tree mixup augmentation dataset
                    mix_label = batch['labels']
                    del batch['labels']

                    outputs = model(**batch)
                    logits = outputs.logits

                    loss = cross_entropy(logits, mix_label)
                else:
                    # train on 00&11 tree mixup augmentation dataset
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                # normal train
                
                batch = {k: v.to(args.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
            # ----------------------------------------------
            #               Aug_dataloader
            # ----------------------------------------------
            if args.mode == 'raw_aug':
                aug_batch = next(aug_dataloader)
                aug_batch = {k: v.to(args.device) for k, v in aug_batch.items()}
                
                if args.mix:
                    mix_label = aug_batch['labels']
                    del aug_batch['labels']
                    aug_outputs = model(**aug_batch)
                    aug_logits = aug_outputs.logits

                    aug_loss = cross_entropy(aug_logits, mix_label)
                else:
                    aug_outputs = model(**aug_batch)
                    aug_loss = aug_outputs.loss
                loss += aug_loss*args.augweight  # for sst2,rte reaches best performance

            # Backward propagation
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if args.local_rank == 0 or args.local_rank == -1:
                writer.add_scalar("Loss/loss", loss, step +
                                  epoch*len(train_dataloader))
                writer.flush()
                if args.random_mix:
                    bar.set_description(
                        '| Epoch: {:<2}/{:<2}| Best acc:{:.2f}| Fail:{}|'.format(epoch, args.epoch, best_acc*100, fail))
                else:
                    bar.set_description(
                        '| Epoch: {:<2}/{:<2}| Best acc:{:.2f}|'.format(epoch, args.epoch, best_acc*100))

            # =================================================
            #                     Validation
            # =================================================
            if (epoch*len(train_dataloader)+step+1) % args.val_steps == 0:
                total_eval_accuracy = 0
                total_val_loss = 0
                model.eval()  # evaluation after each epoch
                for i, batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        batch = {k: v.to(args.device)
                                 for k, v in batch.items()}
                        outputs = model(**batch)
                        logits = outputs.logits
                        loss = outputs.loss

                        if args.n_gpu > 1:
                            loss = loss.mean()
                        logits = logits.detach().cpu().numpy()
                        label_ids = batch['labels'].to('cpu').numpy()

                        accuracy = flat_accuracy(logits, label_ids)
                        if args.local_rank != -1:
                            torch.distributed.barrier()
                            reduced_loss = reduce_tensor(loss, args)
                            accuracy = torch.tensor(accuracy).to(args.device)
                            reduced_acc = reduce_tensor(accuracy, args)
                            total_val_loss += reduced_loss
                            total_eval_accuracy += reduced_acc
                        else:
                            total_eval_accuracy += accuracy.item()
                            total_val_loss += loss.item()
                avg_val_loss = total_val_loss/len(val_dataloader)
                avg_val_accuracy = total_eval_accuracy/len(val_dataloader)
                if avg_val_accuracy > best_acc:
                    best_acc = avg_val_accuracy
                    bset_steps = (epoch*len(train_dataloader) +
                                  step)*args.batch_size
                    if args.save_model:
                        torch.save(model.state_dict(), 'best_model.pt')
                if args.local_rank == 0 or args.local_rank == -1:
                    writer.add_scalar("Test/Loss", avg_val_loss,
                                      epoch*len(train_dataloader)+step)
                    writer.add_scalar(
                        "Test/Accuracy", avg_val_accuracy, epoch*len(train_dataloader)+step)
                    writer.flush()
                    # print(f'Validation loss: {avg_val_loss}')
                    # print(f'Accuracy: {avg_val_accuracy:.5f}')
                    # print('Best Accuracy:{:.5f} Steps:{}\n'.format(best_acc, bset_steps))
    
    if args.data_path:
        aug_num=args.data_path.split('_')[-1]
        
        if args.low_resource_dir:
            # low resource raw_aug
            partial = re.findall(r'low_resource_(0.\d+)',
                                 args.low_resource_dir)[0]
            aug_num_seed = aug_num+'_'+str(args.seed)
            result_logger.info('-'*160)
            result_logger.info('| Data : {} | Mode: {:<8} | #Aug {:<6} | Best acc:{} | Steps:{} | Weight {} |Aug data: {}'.format(
                args.data+'_'+partial, args.mode, aug_num_seed, round(best_acc*100, 3), bset_steps, args.augweight, args.data_path))
        else:
            # raw_aug
            aug_data_seed=re.findall(r'seed(\d)',args.data_path)[0]
            aug_num_seed = aug_num+'_'+aug_data_seed
            result_logger.info('-'*160)
            result_logger.info('| Data : {} | Mode: {:<8} | #Aug {:<6} | Best acc:{} | Steps:{} | Weight {} |Aug data: {}'.format(
            args.data, args.mode, aug_num_seed ,round(best_acc*100,3), bset_steps, args.augweight,args.data_path))
    else:
        if args.low_resource_dir:
            # low resource raw
            partial=re.findall(r'low_resource_(0.\d+)',args.low_resource_dir)[0]
            result_logger.info('-'*160)
            result_logger.info('| Data : {} | Mode: {:.8} | Seed: {} | Best acc:{} | Steps:{} | Randommix: {} | Aug data: {}'.format(
                args.data+'-'+partial, args.mode, args.seed, round(best_acc*100,3), bset_steps,bool(args.random_mix) ,args.data_path))
        else:
            # raw
            result_logger.info('-'*160)
            result_logger.info('| Data : {} | Mode: {:.8} | Seed: {} | Best acc:{} | Steps:{} | Randommix: {} | Aug data: {}'.format(
                args.data, args.mode, args.seed, round(best_acc*100,3), bset_steps, bool(args.random_mix),args.data_path))





def main(args):
    set_seed(args.seed)
    if args.mode in ['raw', 'raw_aug', 'aug']:
        if args.low_resource_dir:
            print("="*20, ' Lowresource ', '='*20)
        train(args)
if __name__ == '__main__':
    args = parse_argument()
    main(args)

import pandas as pd
import text_clean
import random 
import torch
from torch.utils.data import TensorDataset,DataLoader,random_split,RandomSampler
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.nn.parallel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification,AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score,accuracy_score
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import re
from itertools import cycle
import argparse
import torch.distributed as dist
import datetime
import benepar,spacy
import augmentation
import time

def text2tensor(text):
    t=re.findall(r'\d+(\.\d+)?',text)
    pos=float(t[1])
    neg=float(t[0])
    #print(pos,neg)
    return [pos,neg]

def cross_entropy(logits,target):
    p=F.softmax(logits,dim=1)
    log_p=-torch.log(p)
    loss=target*log_p
    #print(target,p,log_p,loss)
    batch_num=logits.shape[0]
    return loss.sum()/batch_num

def encode_fn(text_list,tokenizer):
    """
    transform string to tensor
    input:
        text_list --- list of string 
    output:
        all_input_ids --- tensor of tensors [batch,max_length]
    
    """
    all_input_ids=[]
    for text in text_list:
        input_ids=tokenizer.encode(text,add_special_tokens=True,max_length=120,padding='max_length',return_tensors='pt',truncation=True)
        all_input_ids.append(input_ids)
    all_input_ids=torch.cat(all_input_ids,dim=0)
    return all_input_ids
def random_span_exchange(args,ids1,lam):
    """
    function: random select span to exchange based on lam to decide span length and rand_index decide selected candidate exchange sentece
    input:
        ids1 -- tensors of tensors input_ids 
        lam  -- sample exchange rate
    output:
        ids1 -- tensors of tensors , exchanged span
        rand_index -- tensors , permutation index
    
    """
    rand_index=torch.randperm(ids1.shape[0])
    lenlist=[]
    rand_index=torch.randperm(len(ids1))
    for x in ids1:
        mask=((x!=101)&(x!=0)&(x!=102))
        lenlist.append(int(mask.sum()))
    lenlist2=torch.tensor(lenlist)[rand_index]
    spanlen=torch.tensor([int(x*lam) for x in lenlist])

    beginlist=[1+random.randint(0,x-int(x*lam)) for x in lenlist]
    beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen)]
    if args.difflen:
        spanlen2=torch.tensor([int(x*lam) for x in lenlist2])
        spanlist2=[(x,int(y)) for x,y in zip(beginlist2,spanlen2)]
    else:
        spanlist2=[(x,int(y)) for x,y in zip(beginlist2,spanlen)]
    spanlist=[(x,int(y)) for x,y in zip(beginlist,spanlen)]
    
    ids2=ids1.clone()
    if args.difflen:
        for idx in range(len(ids1)):    
            tmp=torch.cat((ids1[idx][:spanlist[idx][0]],ids2[rand_index[idx]][spanlist2[idx][0]:spanlist2[idx][0]+spanlist2[idx][1]],ids1[idx][spanlist[idx][0]+spanlist[idx][1]:]),dim=0)[:ids1.shape[1]]
            ids1[idx]=torch.cat((tmp,torch.zeros(ids1.shape[1]-len(tmp))))
    else:      
        for idx in range(len(ids1)):
            ids1[idx][spanlist[idx][0]:spanlist[idx][0]+spanlist[idx][1]]=ids2[rand_index[idx]][spanlist2[idx][0]:spanlist2[idx][0]+spanlist2[idx][1]]
    assert ids1.shape==ids2.shape
    return ids1,rand_index
def flat_accuracy(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return accuracy_score(labels_flat,pred_flat)
def reduce_tensor(tensor,args):
    rt=tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM) 
    rt/=args.world_size
    return rt
def augment(args,source_batch,nlp,model,tokenizer):
    """
    This function is to generate augmentation dataset based on source_batch
    
    input:
        source_batch -- batch to be augmented , contains pos&neg samples with binary labels
    return:
        aug_batch -- batches of augmentation data, contains with a two dimensional soft labels
    """
    sentence=[] # list of str
    labels=[]   # list tensor
    all_samples=list(zip(source_batch[0],source_batch[1]))
    # print('all_sample ',all_samples)
    pos_samples=[sample[0] for sample in all_samples if sample[1]==1 and len(str(sample[0]).split())>args.token_num]
    neg_samples=[sample[0] for sample in all_samples if sample[1]==0 and len(str(sample[0]).split())>args.token_num]
    # pp_pairs=[random.sample(pos_samples,2)[0] for _ in range(args.pp)]
    # nn_pairs=[random.sample(neg_samples,2)[0] for _ in range(args.nn)]
    pn_pairs=[(random.sample(pos_samples,1)[0],random.sample(neg_samples,1)[0]) for _ in range(args.pairs//2)]
    with torch.no_grad():
        for (pos_sample,neg_sample) in pn_pairs:
            s1,s2=augmentation.exchange_span(args,pos_sample,neg_sample,nlp,model=model,tokenizer=tokenizer)
            if s1: #in case s1 or s2 is None
                sentence.append(s1[0])
                sentence.append(s2[0])
                labels.append(s1[1])
                labels.append(s2[1])
   
    sentence=encode_fn(sentence,tokenizer) # transform list of string to tensor of tensor [batch,seq]
    labels=torch.stack(labels,dim=0) # transform list of tensors to tensor of tensors [batch,2]
    return sentence,labels
            
    
    
def train(args,model,tokenizer):
    
    print('='*20,'Start processing dataset','='*20)
    t1=time.time()
    
    # loading training data
    for file in os.listdir(args.data_dir):
        if re.findall('train',file):
            args.train_path=os.path.join(args.data_dir,file)
        if re.findall('dev',file):
            args.dev_path=os.path.join(args.data_dir,file)
    if args.train_path.split('.')[-1]=='tsv':
        sst_train=pd.read_csv(args.train_path,sep='\t')
    else:
        sst_train=pd.read_csv(args.train_path)
        
    if args.dev_path.split('.')[-1]=='tsv':
        sst_dev=pd.read_csv(args.dev_path,sep='\t')
    else:
        sst_dev=pd.read_csv(args.dev_path)
   
   
    if args.train_online:
        sst_train=sst_train.reindex(sst_train['sentence'].str.len().sort_values(ascending=False).index)
    
    train_input_ids=encode_fn(sst_train['sentence'].values,tokenizer)
    train_labels=torch.tensor(sst_train['label'].values)
    val_input_ids=encode_fn(sst_dev['sentence'].values,tokenizer)
    val_labels=torch.tensor(sst_dev['label'].values)    
    
    train_dataset=TensorDataset(train_input_ids,train_labels)
    val_dataset=TensorDataset(val_input_ids,val_labels)
    if args.local_rank==-1:
        train_sampler=RandomSampler(train_dataset)
        val_sampler=RandomSampler(val_dataset)
    else:
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset) 

    if args.train_online:
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False)
    else:      
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,sampler=train_sampler)
    val_dataloader=DataLoader(val_dataset,batch_size=args.batch_size,sampler=val_sampler)
    t2=time.time()
    
    # unfinished
    # if args.train_aug:
    #     sst_aug=pd.read_csv('SST-2/aug_attention.csv',index_col=0)
    #     sst_aug['labels']=sst_aug['1'].apply(text2tensor)
    #     aug_values=sst_aug['0'].values
    #     aug_labels=sst_aug['labels'].values
    #     aug_labels_list=list(aug_labels)
    #     aug_input_ids=encode_fn(aug_values,tokenizer)
    #     aug_labels=torch.tensor(aug_labels_list)
    #     aug_dataset=TensorDataset(aug_input_ids,aug_labels)
    #     if args.local_rank==-1:
    #         aug_sampler=RandomSampler(aug_dataset)
    #     else:
    #         aug_sampler=torch.utils.data.distributed.DistributedSampler(aug_dataset)
    #     aug_dataloader=DataLoader(aug_dataset,batch_size=args.batch_size//4,sampler=aug_sampler)
    #     aug_dataloader=cycle(aug_dataloader)
        
    print('='*20,'Dataset process done! cost {:.2f}s'.format(t2-t1),'='*20)
    
    print('='*20,'Train dataset path: {}'.format(args.train_path),'='*20)
    
    if args.train_online:
        print('='*20,'Train online ...','='*20)
    if args.train_aug:
        print('='*20,'Augmentation dataset path: {}'.format(args.train_path),'='*20)
    
    
 
    
   
    optimizer=AdamW(model.parameters(),lr=2e-5)
    scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=args.epoch*len(train_dataloader))
    criterion=nn.CrossEntropyLoss()
    model.train()
    if args.local_rank==0 or args.local_rank==-1:
        if args.train:
            writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'Train_{}'.format(datetime.datetime.now())))
            
        elif args.train_online:
            if args.difflen:
                if args.onlyaug:
                    writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'diff_online_onlyaug_{}'.format(args.alpha)))
                else:
                    writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'diff_online_all_{}'.format(args.alpha)))
            else:
                if args.onlyaug:
                    writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'online_onlyaug_{}'.format(args.alpha)))
                else:
                    writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'online_all_{}'.format(args.alpha)))
                
        elif args.train_aug:
            writer=SummaryWriter(log_dir=os.path.join(args.data_dir,'trainaug_{}'.format(datetime.datetime.now())))
            
    print('='*20,'Start training','='*20)
    
    for epoch in range(args.epoch):
        bar=tqdm(enumerate(train_dataloader),total=len(train_dataloader)//args.world_size)
        online_fail=0
        for step,batch in bar:   
            model.zero_grad() 
            if args.train_online:
                try:
                    input_ids,target_a=batch[0],batch[1]
                    lam=np.random.beta(10,10)
                    lam=args.alpha
                    exchanged_ids,rand_index=random_span_exchange(args,input_ids,lam)
                    target_b=target_a[rand_index]
                    # print(exchanged_ids)
                    outputs=model(exchanged_ids.to(args.device),token_type_ids=None,attention_mask=(exchanged_ids>0).to(args.device))  
                    logits=outputs.logits
                    # print(logits,target_a,target_b,lam)
                    loss=criterion(logits,target_a.to(args.device))*(1-lam)+criterion(logits,target_b.to(args.device))*lam
                    # print(loss)
                except Exception as e:
                    print(e)
                    online_fail+=1
                    print('online_augmentation_failed {} times'.format(online_fail))
                    if args.onlyaug:
                        continue
                    outputs= model(batch[0].to(args.device),token_type_ids=None,attention_mask=(batch[0]>0).to(args.device),labels=batch[1].to(args.device))
                    loss=outputs.loss                   
            if args.train:
                outputs= model(batch[0].to(args.device),token_type_ids=None,attention_mask=(batch[0]>0).to(args.device),labels=batch[1].to(args.device))
                loss=outputs.loss  
                # print(loss)     
            if args.n_gpu>1:
                loss=loss.mean()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if args.local_rank==0 or args.local_rank==-1:      
                writer.add_scalar("Loss/loss",loss,step+epoch*len(train_dataloader))
                writer.flush()
                bar.set_description('Train loss:{}'.format(loss))
        total_eval_accuracy =0
        total_val_loss=0
        model.eval() #evaluation after each epoch
        for i ,batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(batch[0].to(args.device),token_type_ids=None, attention_mask=(batch[0]>0).to(args.device), labels=batch[1].to(args.device))
                logits=outputs.logits
                loss=outputs.loss
                    
                if args.n_gpu>1:
                    loss=loss.mean()
                logits=logits.detach().cpu().numpy()
                label_ids=batch[1].to('cpu').numpy()
                
                accuracy=flat_accuracy(logits,label_ids)
                if args.local_rank!=-1:
                    torch.distributed.barrier()
                    reduced_loss=reduce_tensor(loss,args)
                    accuracy=torch.tensor(accuracy).to(args.device)
                    reduced_acc=reduce_tensor(accuracy,args)
                    total_val_loss+=reduced_loss
                    total_eval_accuracy+=reduced_acc
                else:
                    total_eval_accuracy+=accuracy.item()
                    total_val_loss+=loss.item()
                    
        avg_val_loss=total_val_loss/len(val_dataloader)
        avg_val_accuracy=total_eval_accuracy/len(val_dataloader)
        
    
        if args.local_rank==0 or args.local_rank==-1:
            writer.add_scalar("Test/Loss",avg_val_loss,epoch)
            writer.add_scalar("Test/Accuracy",avg_val_accuracy,epoch)
            writer.flush()
            print(f'Validation loss: {avg_val_loss}')
            print(f'Accuracy: {avg_val_accuracy:.5f}')
            print('\n')

    # torch.save(model.state_dict(),'SST-2/best_model.pt')
def main():
    parser=argparse.ArgumentParser()
    # parser.add('--data',metaval='dir',help='path to dataset',required=True)
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--train_aug',action='store_true',help="train model on aug data or not")
    parser.add_argument('--train',action='store_true',help="train model or not")
    parser.add_argument('--train_online',action='store_true',help="train model or not")
    
    
    #training
    parser.add_argument('--data_dir',type=str,required=True,help="data path")
    parser.add_argument('--epoch',type=int,default=5,help='train epochs')
    parser.add_argument('--batch_size',default=128,type=int,help='train examples in each batch')
     
    # online random
    parser.add_argument('--alpha',type=float,default=0.1,help="online augmentation alpha")
    parser.add_argument('--onlyaug',action='store_true',help="train only on online aug batch")
    parser.add_argument('--difflen',action='store_true',help="train only on online aug batch")
    
    
    # labeling
    parser.add_argument('--tfidf',action='store_true',help='labels weight on tfidf score')
    parser.add_argument('--attention',action='store_true',help='labels weight on attention score')
    parser.add_argument('--token_num',type=int,default=10,help='mininum number of tokens in sentence chosen to augment')
    parser.add_argument('--show_info',type=int,default=10,help='times to print details of augmentation')
   
    parser.add_argument('--val_steps',default=50,type=int,help='evaluate on dev datasets every steps')
    parser.add_argument('--pairs',default=256,type=int,help='augmentation samples in each batch')
    
    args=parser.parse_args()
    
    t1=time.time()
    print('='*20,'Loading models','='*20)
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1 #the number of gpu on each proc
    
    args.device=device

    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

    # device=torch.device('cuda')
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    if args.local_rank!=-1:
        args.world_size=torch.cuda.device_count()
    else:
        args.world_size=1
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    model.to(device)
    t2=time.time()
    print('='*20,'Loading models complete!, cost {:.2f}s'.format(t2-t1),'='*20)
    if args.local_rank!=-1:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
    elif args.n_gpu>1:
        model=nn.DataParallel(model)

    if args.train or args.train_aug or args.train_online:
         train(args,model,tokenizer)
if __name__=='__main__':
    main()
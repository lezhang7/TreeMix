import torch
import random
import numpy as np
def random_mixup_process(args,ids1,lam):
    
    rand_index=torch.randperm(ids1.shape[0])
    lenlist=[]
    # rand_index=torch.randperm(len(ids1))
    for x in ids1:
        mask=((x!=101)&(x!=0)&(x!=102))
        lenlist.append(int(mask.sum()))
    lenlist2=torch.tensor(lenlist)[rand_index]
    spanlen=torch.tensor([int(x*lam) for x in lenlist])

    beginlist=[1+random.randint(0,x-int(x*lam)) for x in lenlist]
    beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen)]
    if args.difflen:
    
        spanlen2=torch.tensor([int(x*lam) for x in lenlist2])
        # beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen2)]
        spanlist2=[(x,int(y)) for x,y in zip(beginlist2,spanlen2)]
    else:
        # beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen)]
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
def mixup_01(args,input_ids,lam,idx1,idx2):
    '''
    01交换
    '''
    difflen=False
    random_index=torch.zeros(len(idx1)+len(idx2)).long()
    random_index[idx1]=torch.tensor(np.random.choice(idx2,size=len(idx1)))
    random_index[idx2]=torch.tensor(np.random.choice(idx1,size=len(idx2)))
    
    len_list1=[]
    len_list2=[]
    for input_id1 in input_ids:
        #计算各个句子的具体token数
        mask=((input_id1!=101)&(input_id1!=0)&(input_id1!=102))
        len_list1.append(int(mask.sum()))
    # print(len_list1)
    len_list2=torch.tensor(len_list1)[random_index]
    
    spanlen=torch.tensor([int(x*lam) for x in len_list1])
    beginlist=[1+random.randint(0,x-int(x*lam)) for x in len_list1]
    beginlist2=[1+random.randint(0,x-y) for x,y in zip(len_list2,spanlen)]
    if difflen:
        spanlen2=torch.tensor([int(x*lam) for x in len_list2])
        # beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen2)]
        spanlist2=[(x,int(y)) for x,y in zip(beginlist2,spanlen2)]
    else:
        # beginlist2=[1+random.randint(0,x-y) for x,y in zip(lenlist2,spanlen)]
        spanlist2=[(x,int(y)) for x,y in zip(beginlist2,spanlen)]
    spanlist=[(x,int(y)) for x,y in zip(beginlist,spanlen)]
    new_ids=input_ids.clone()
    
    # print(random_index)
    if difflen:
        for idx in range(len(ids1)):    
            tmp=torch.cat((ids1[idx][:spanlist[idx][0]],ids2[rand_index[idx]][spanlist2[idx][0]:spanlist2[idx][0]+spanlist2[idx][1]],ids1[idx][spanlist[idx][0]+spanlist[idx][1]:]),dim=0)[:ids1.shape[1]]
            ids1[idx]=torch.cat((tmp,torch.zeros(ids1.shape[1]-len(tmp))))
    else:      
        for idx in range(len(input_ids)):
            new_ids[idx][spanlist[idx][0]:spanlist[idx][0]+spanlist[idx][1]]=input_ids[random_index[idx]][spanlist2[idx][0]:spanlist2[idx][0]+spanlist2[idx][1]]
    # for i in range(len(input_ids)):
    #     print('{}:交换的是{}与{}，其中第1句选取的是从{}开始到{}的句子，第2句选取的是从{}开始到{}结束的句子'.format(
    #         i,i,random_index[i],spanlist[i][0],spanlist[i][0]+spanlist[i][1],spanlist2[i][0],spanlist2[i][0]+spanlist2[i][1]))
    return new_ids,random_index 
def mixup(args,input_ids,lam,idx1,idx2=None):
    '''
    只针对idx1索引对应的sample内部进行交换，如果idx2也给的话就是idx1 idx2进行交换 
    '''
    select_input_ids=torch.index_select(input_ids,0,idx1)
    rand_index=torch.randperm(select_input_ids.shape[0])
    new_idx=torch.tensor(list(range(input_ids.shape[0])))
    len_list1=[]
    len_list2=[]
    for input_id1 in select_input_ids:
        #calculte length of tokens in each sentence
        mask=((input_id1!=101)&(input_id1!=0)&(input_id1!=102))
        len_list1.append(int(mask.sum()))
    len_list2=torch.tensor(len_list1)[rand_index]
    
    spanlen=torch.tensor([int(x*lam) for x in len_list1])
    beginlist=[1+random.randint(0,x-y) for x,y in zip(len_list1,spanlen)]
    beginlist2=[1+random.randint(0,max(0,x-y)) for x,y in zip(len_list2,spanlen)]
    
    spanlist=[(x,int(y)) for x,y in zip(beginlist,spanlen)]
    spanlist2 = [(x, min(int(y),z)) for x, y, z in zip(beginlist2, spanlen, len_list2)]
    new_ids=input_ids.clone()
    new_idx[idx1]=idx1[rand_index]
    for i,idx in enumerate(idx1):
        new_ids[idx][spanlist[i][0]:spanlist[i][0]+spanlist[i][1]]=input_ids[idx1[rand_index[i]]][spanlist2[i][0]:spanlist2[i][0]+spanlist2[i][1]]

    return new_ids,new_idx
def random_mixup(args,ids1,lab1,lam):
    """
    function: random select span to exchange based on lam to decide span length and rand_index decide selected candidate exchange sentece
    input:
        ids1 -- tensors of tensors input_ids 
        lab1 -- tensors of tensors labels
        lam  -- span length rate
    output:
        ids1 -- tensors of tensors , exchanged span
        rand_index -- tensors , permutation index
    
    """
    if args.random_mix=='all':
        return mixup(args,ids1,lam,torch.tensor(range(ids1.shape[0])))
    else:
        pos_idx=(lab1==1).nonzero().squeeze()
        neg_idx=(lab1==0).nonzero().squeeze()
        pos_samples=torch.index_select(ids1,0,pos_idx)
        neg_samples=torch.index_select(ids1,0,neg_idx)
        if args.random_mix=='zero':
            return mixup(args,ids1,lam,neg_idx)
        if args.random_mix=='one':
            return mixup(args,ids1,lam,pos_idx)
        if args.random_mix=='zero_one':
            return mixup_01(args,ids1,lam,pos_idx,neg_idx)  

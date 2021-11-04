import random
from nltk import Tree
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import os
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
def subtree_exchange(args,parsing1,label1,parsing2,label2,lam1,lam2):
    """
    For a pair sentence, exchange subtree and return a label based on subtree length
     
    Find the candidate subtree, and extract correspoding span, and exchange span
    
    """
    assert lam1>lam2
    t1=Tree.fromstring(parsing1)
    original_sentence=' '.join(t1.leaves())
    t1_len=len(t1.leaves())
    candicate_subtree1=list(t1.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))
    t2=Tree.fromstring(parsing2)
    candicate_subtree2=list(t2.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))
    
    # print('subtree1:',len(candicate_subtree1),'\nsubtree2:',len(candicate_subtree2))
    if len(candicate_subtree1)==0 or len(candicate_subtree2)==0:
        # print("this pair fail",len(candicate_subtree1),len(candicate_subtree2))
        return None
    
    if args.same_type:
        tree_labels1=[tree.label() for tree in candicate_subtree1]
        tree_labels2=[tree.label() for tree in candicate_subtree2]
        same_labels=list(set(tree_labels1)&set(tree_labels2))
        if not same_labels:
            # print('无相同类型的子树')
            return None
        select_label=random.choice(same_labels)
        candicate1=random.choice([t for t in candicate_subtree1 if t.label()==select_label])
        candicate2=random.choice([t for t in candicate_subtree2 if t.label()==select_label])
    else:
        candicate1=random.choice(candicate_subtree1) 
        candicate2=random.choice(candicate_subtree2) 
        
    exchanged_span=' '.join(candicate1.leaves())
    exchanged_len=len(candicate1.leaves())
    exchanging_span=' '.join(candicate2.leaves())
    new_sentence=original_sentence.replace(exchanged_span,exchanging_span)
    if label1!=label2:
        exchanging_len=len(candicate2.leaves())
        new_len=t1_len-exchanged_len+exchanging_len
        new_label=(exchanging_len/new_len)*label2+(new_len-exchanging_len)/new_len*label1
    else:
        new_label=label1
    # print('被替换的span:{}\n替换的span:{}'.format(exchanged_span,exchanging_span))
    return new_sentence,new_label
def augmentation(args,dataset,aug_times,lam1,lam2):
    """
    generate aug_num augmentation dataset 
    input:
        dataset --- pd.dataframe
    output:
        aug_dataset --- pd.dataframe
    """
    generated_list=[]
    data_list=dataset.values.tolist()
    shuffled_list=data_list.copy()
    with tqdm(total=aug_times*len(data_list)) as bar:
        for i in range(aug_times):
            np.random.shuffle(shuffled_list)
            for idx in range(len(data_list)):
                bar.update(1)
                aug_sample=subtree_exchange(args,data_list[idx][2],data_list[idx][1],shuffled_list[idx][2],shuffled_list[idx][1],lam1,lam2)
                if  aug_sample: 
                    generated_list.append(aug_sample)
    #De-duplication
    generated_list=list(set(generated_list))
    return generated_list
def main():
    parser=argparse.ArgumentParser() 
    parser.add_argument('--attention',action='store_true',help='labels weight on attention score')
    parser.add_argument('--lam1',type=float,default=0.6)
    parser.add_argument('--lam2',type=float,default=0.3)
    parser.add_argument('--times',type=int,default=5)
    parser.add_argument('--min_token',type=int,default=10,help='minimum token numbers of augmentation samples')
    parser.add_argument('--same_type',action='store_true')
    parser.add_argument('--seed',default=7,type=int)
    # parser.add_argument('--data_path',type=str,required=True)
    parser.add_argument('--output_dir',type=str,required=True)
    
    
    # parser.add_argument('--load_path',metavar='dir',required=True,help='directory of created augmentation dataset')
    args=parser.parse_args()
    set_seed(args.seed)
    dataset=pd.read_csv("SST-2/train_parsing.csv")
    if args.min_token:
        dataset=dataset.loc[dataset['sentence'].str.split().apply(lambda x:len(x)>args.min_token)]
    pos_samples=dataset.loc[dataset["label"]==1]
    neg_samples=dataset.loc[dataset["label"]==0]
    pos_pd=pd.DataFrame(augmentation(args,pos_samples,args.times,args.lam1,args.lam2),columns=["sentence","label"])
    neg_pd=pd.DataFrame(augmentation(args,neg_samples,args.times,args.lam1,args.lam2),columns=["sentence","label"])
    new_pd=pd.concat([pos_pd,neg_pd],axis=0)
    new_pd=new_pd.sample(frac=1)
    if args.same_type:
        new_pd.to_csv(os.path.join(args.output_dir,'sametype_generated_times{}_seed{}_{}_{}_{}k.csv'.format(args.times,args.seed,args.lam1,args.lam2,round(len(new_pd))//1000,-1)),index=0)
    else:   
        new_pd.to_csv(os.path.join(args.output_dir,'generated_times{}_seed{}_{}_{}_{}k.csv'.format(args.times,args.seed,args.lam1,args.lam2,round(len(new_pd))//1000,-1)),index=0)
    
    
    
if __name__=='__main__':
    main()
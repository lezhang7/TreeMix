import benepar,spacy
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import numpy as np
import torch.nn.functional as F
import random
import torch
import re
from tqdm import tqdm
import argparse


def get_weightedlist(sparse_result,text_list,vocab):
    """
    input: 
        text --- list of str or str
        spare_result --- matrix of #str * #word
    output:
        score_list --- list of score list of each word in sentence
    """
    if type(text_list)==str:
        text_list=[text_list]
    score_list=[]
    for i,text in enumerate(text_list):
        text_list=text.split()
        score=[]
        for token in text_list:
            try:
                score.append(sparse_result[i][vocab[token]])
            except:
                score.append(0)
        score_list.append(score)
    return score_list
def tfidf_weight(full_text,exchanged_span,vectorizer,vocab):
    words_score=get_weightedlist(vectorizer.transform([full_text]).toarray(),full_text,vocab)[0]
    words_score=np.array(words_score)
    position=position_match(full_text,exchanged_span)
    scores=(words_score*position).sum()
    weight=scores/(words_score.sum())
    #print(words_score,'\n',position,weight)
    return round(weight,2)


def attention_weight(full_text,exchanged_span,tokenizer,model):
    input=tokenizer(full_text,return_tensors='pt')
    for i in input:
        input[i]=input[i].cuda()
    output=model(**input,output_attentions=True)
    last_layer_attention=output.attentions[-1].squeeze()
    words_score=F.softmax(last_layer_attention.sum(dim=0)[0][1:-1],dim=0).cpu().numpy()
    position=attention_position_match(full_text,exchanged_span,tokenizer)
    # print(tokenizer.tokenize(full_text),'\n',full_text,full_text.split(),'\n',last_layer_attention.shape,position.shape,words_score.shape)
    scores=(words_score*position).sum()
    weight=scores/(words_score.sum())
    return round(weight,2)
def attention_position_match(s,span,tokenizer):
    s=tokenizer.tokenize(s)
    span=tokenizer.tokenize(span)
    num=0
    num_max=len(span)
    index_list=[]
    #print(s,span)
    for index in range(len(s)):
        if s[index]==span[num]:
            index_list.append(1)
            num+=1
        else:
            index_list.append(0)
        if num==num_max:
            break
    index_list+=(len(s)-index-1)*[0]

    return np.array(index_list)   
def position_match(s,span):
    
    s=s.split()
    span=span.split()
    num=0
    num_max=len(span)
    index_list=[]
    #print(s,span)
    for index in range(len(s)):
        if s[index]==span[num]:
            index_list.append(1)
            num+=1
        else:
            index_list.append(0)
        if num==num_max:
            break
    index_list+=(len(s)-index-1)*[0]

    return np.array(index_list)

def findall_VPNP(raw_text,nlp):
    """"
    input: 
        raw_text --- str
        nlp --- benepar parse model
    output:
        vp_list --- list contains all possible vp part
        np_list --- list contains all possible np part
    """
    doc=nlp(raw_text)
    sent=list(doc.sents)[0]
    vp_list=[]
    np_list=[]
    def find_VP_NP(children): 
        """
        input: list of spacy.tokens.span
        """
        #print(children)
        for i in children:
            #print(i,i._.labels)
            if 'VP' in i._.labels:
                vp_list.append(str(i))
            if 'NP' in i._.labels:
                np_list.append(str(i))
            find_VP_NP(i._.children)
    find_VP_NP(sent._.children)
    #print('vp_parts:',vp_list,'\nnp_parts:',np_list,'\n\n')
    return vp_list,np_list
def exchange_span(args,pos_example,neg_example,nlp,vectorizer=None,vocab=None,model=None,tokenizer=None):
    """"
    如果两个句子都有vp,随机选一个vp进行交换
    如果两个句子有一个句子vp是空列表，那么就在两个的np中随机交换
    如果两个np也有一个为0，那就跳过这对
    
    """
    
    pos_vp_list,pos_np_list=findall_VPNP(pos_example,nlp)
    neg_vp_list,neg_np_list=findall_VPNP(neg_example,nlp)
    if pos_vp_list and neg_vp_list:
        # print('Will change vp part!')
        candicate_span1=random.choice(pos_vp_list)
        candicate_span2=random.choice(neg_vp_list)
    elif pos_np_list and neg_np_list:
        candicate_span1=random.choice(pos_np_list)
        candicate_span2=random.choice(neg_np_list)
    else:
        return None,None
    
    if args.tfidf:
        weight1=tfidf_weight(pos_example,candicate_span1,vectorizer,vocab) # pos span replaced by neg one
        weight2=tfidf_weight(neg_example,candicate_span2,vectorizer,vocab) # neg span replaced by pos one
    elif args.attention:
        weight1=attention_weight(pos_example,candicate_span1,tokenizer,model)
        weight2=attention_weight(neg_example,candicate_span2,tokenizer,model)
        
    label1=F.softmax(torch.tensor([weight2,1-weight1]),dim=0)
    label2=F.softmax(torch.tensor([1-weight2,weight1]),dim=0)
   
    exchanged_example1=re.sub(candicate_span1,candicate_span2,pos_example)
    exchanged_example2=re.sub(candicate_span2,candicate_span1,neg_example)
    
    if args.show_info:
        print('\nbefore exchange\npos:\t{}\nneg:\t{}'.format(pos_example,neg_example))
        print('candicate1:\t{}\tweight:{}\ncandicate2:\t{}\tweight{}'.format(candicate_span1,weight1,candicate_span2,weight2))
        print('after exchange\ns1:\t{}\t{}\ns2:\t{}\t{}'.format(exchanged_example1,label1,exchanged_example2,label2))
        args.show_info-=1
    return (exchanged_example1,label1),(exchanged_example2,label2)

def main():
    sst_train=pd.read_csv('SST-2/train.tsv',sep='\t')
    parser=argparse.ArgumentParser()
    parser.add_argument('--tfidf',action='store_true',help='labels weight on tfidf score')
    parser.add_argument('--attention',action='store_true',help='labels weight on attention score')
    parser.add_argument('--token_num',type=int,default=10,help='mininum number of tokens in sentence chosen to augment')
    parser.add_argument('--aug_num',type=int,default=10000,help='number of augmentation samples')
    parser.add_argument('--show_info',action='store_true',help='whther to print details of augmentation')
    
    # parser.add_argument('--load_path',metavar='dir',required=True,help='directory of created augmentation dataset')
    args=parser.parse_args()
    # load data
    corpus=sst_train['sentence'].tolist()
    
    # load label weight tool
    if args.tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print('create data based on tfidf')
        vectorizer=TfidfVectorizer()
        vectorizer.fit(corpus)
        vocab=vectorizer.vocabulary_ 
        aug_dir="./SST-2/aug_tfidf.csv"
    elif args.attention:
        from transformers import BertTokenizer
        from transformers import BertForSequenceClassification
        model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        model=model.cuda()
        aug_dir="./SST-2/aug_attention.csv"
    # load constituency parser tool
    nlp=spacy.load('en_core_web_md')
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
   
    # prepare candidate dataset
    z=zip(list(sst_train['sentence']),list(sst_train['label']))
    pos_data=[s[0] for s in z if s[1]==1 and len(str(s[0]).split())>args.token_num]
    random.shuffle(pos_data)
    z=zip(list(sst_train['sentence']),list(sst_train['label']))
    neg_data=[s[0] for s in z if s[1]==0 and len(str(s[0]).split())>args.token_num]
    random.shuffle(neg_data)
    pair=list(zip(pos_data[:args.aug_num],neg_data[:args.aug_num]))
    
    # create augmentation dataset!
    aug_list=[]
    failcnt=0
    with torch.no_grad():
        
        for step,(pos_example,neg_example) in tqdm(enumerate(pair),total=len(pair)):
            # try:
            #     if args.tfidf:
            #         aug_example1,aug_example2=exchange_span(args,pos_example,neg_example,nlp,vectorizer=vectorizer,vocab=vocab)
            #     elif args.attention:
            #         aug_example1,aug_example2=exchange_span(args,pos_example,neg_example,nlp,tokenizer=tokenizer,model=model)
                
            #     if aug_example1:
            #         aug_list.append(aug_example1)
            #         aug_list.append(aug_example2)
            # except Exception as e:
            #     failcnt+=1
            #     print('error encounter {}'.format(e))
            #     print('one example change fail! fail {} times!'.format(failcnt))
            try:
                if step>5:
                    args.show_info=False
                if args.tfidf:
                    aug_example1,aug_example2=exchange_span(args,pos_example,neg_example,nlp,vectorizer=vectorizer,vocab=vocab)
                elif args.attention:
                    aug_example1,aug_example2=exchange_span(args,pos_example,neg_example,nlp,tokenizer=tokenizer,model=model)
                if aug_example1:
                    aug_list.append(aug_example1)
                    aug_list.append(aug_example2)  
                args.show_info+=1 
            except Exception as e:
                failcnt+=1
                print('error encounter {}'.format(e))
                print('one example change fail! fail {} times!'.format(failcnt))
                    
        aug_pd=pd.DataFrame(aug_list)
        aug_pd.to_csv(aug_dir)
      
         
    
if __name__=='__main__':
    main()
    
   
    
    
import random
from nltk import Tree
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import os
from datasets import load_dataset, Dataset, concatenate_datasets
from process_data import settings
from multiprocessing import Pool,cpu_count


def modify(commands, debug=0):
    commands = commands.split(' ')
    verb = ['look', 'jump', 'walk', 'turn', 'run']
    end_sign = ['left', 'right', 'twice', 'thrice']
    add_pos = []
    if debug:
        print(commands)
    for i in range(len(commands)-1):
        if commands[i] in end_sign and commands[i+1] in verb:
            add_pos.append(i+1)
            # commands.insert(i+1,'and')
            if debug:
                print(commands)
        if commands[i] in verb and commands[i+1] in verb:
            add_pos.append(i+1)
            # commands.insert(i+1,'and')
            if debug:
                print(commands)
    for i, pos in enumerate(add_pos):
        commands.insert(pos+i, 'and')  
    if debug:
        print(commands)
    return ' '.join(commands)


def c2a(commands, debug=0):
    verb = {'look': 'I_LOOK', 'walk': 'I_WALK',
            'run': 'I_RUN', 'jump': 'I_JUMP'}
    direction = {'left': 'I_TURN_LEFT', 'right': 'I_TURN_RIGHT'}
    times = {'twice': 2, 'thrice': 3}
    conjunction = ['and', 'after']

    commands = commands.split(' ')
    actions = []
    previous_command = []
    pre_actions = []
    flag = 0
    i = 0
    if debug:
        print('raw:', commands)
    while len(commands) > 0:
        current = commands.pop(0)
        if debug:
            print('-'*50)
            print('step ', i)
            i += 1
            print('current command:', current, len(commands))
            print('curret waiting commands list:', previous_command)
            print('already actions:', actions)
            print('previous waiting actions:', pre_actions)
        if current in verb.keys() or current == 'turn' or current in conjunction:  # add new actions
            if current == 'and':
                continue
            if not previous_command:  # initialization
                previous_command.append(current)

            else:  # one conmands over
                if debug:
                    print('##### one commands over#####')
                current_action = translate(previous_command)
                previous_command = []
                if debug:
                    print(
                        '****got new action from previous commandsa list:{}****'.format(current_action[0]))

                if current == 'after':
                    pre_actions.extend(current_action)
                    if debug:
                        print('****this action into pre_actions****')
                elif pre_actions:
                    if debug:
                        print(
                            '****pre_actions and current_actions into action list****')
                    actions.extend(current_action)
                    actions.extend(pre_actions)
                    pre_actions = []
                    previous_command.append(current)
                else:
                    # current is a verb
                    previous_command.append(current)
                    actions.extend(current_action)
        else:
            previous_command.append(current)
    if previous_command:
        current_action = translate(previous_command)
        actions.extend(current_action)
    if pre_actions:
        actions.extend(pre_actions)
    if debug:
        print('-'*50)
        print('over')
        print('previous_command', previous_command)
        print('pre_actions', pre_actions)
        print('current action', current_action)
    return actions


def translate(previous_command):
    verb = {'look': 'I_LOOK', 'walk': 'I_WALK',
            'run': 'I_RUN', 'jump': 'I_JUMP'}
    direction = {'left': 'I_TURN_LEFT', 'right': 'I_TURN_RIGHT'}
    times = {'twice': 2, 'thrice': 3}
    conjunction = ['and', 'after']
    if previous_command[-1] in times.keys():
        return translate(previous_command[:-1])*times[previous_command[-1]]
    if len(previous_command) == 1:
        return [verb[previous_command[0]]]
    elif len(previous_command) == 2:
        if previous_command[0] == 'turn':
            return [direction[previous_command[1]]]
        elif previous_command[1] in direction:
            return [direction[previous_command[1]], verb[previous_command[0]]]
    elif len(previous_command) == 3:
        if previous_command[0] == 'turn':
            if previous_command[1] == 'opposite':
                return [direction[previous_command[2]]]*2
            else:
                return [direction[previous_command[2]]]*4
        elif previous_command[0] in verb.keys():
            if previous_command[1] == 'opposite':
                return [direction[previous_command[2]], direction[previous_command[2]], verb[previous_command[0]]]
            else:
                return [direction[previous_command[2]], verb[previous_command[0]]]*4
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
def subtree_exchange_scan(args,parsing1,parsing2):
    new_sentence=None
    try:
        if args.debug:
            print('check5')
        t1 = Tree.fromstring(parsing1)
        t2 = Tree.fromstring(parsing2)
        t1_len=len(t1.leaves())
        t2_len=len(t2.leaves())
        # ----- restrict label--------------
        # candidate_subtree1=list(t1.subtrees(lambda t:   t.label() in ['VP','VB']))
        # candidate_subtree2 = list(t2.subtrees(lambda t:  t.label() in ['VP', 'VB']))
        # tree_labels1 = [tree.label() for tree in candidate_subtree1]
        # tree_labels2 = [tree.label() for tree in candidate_subtree2]
        # same_labels = list(set(tree_labels1) & set(tree_labels2))
        # if not same_labels:
        #     if args.debug:
        #         print('no same label')
        #     return None
        # select_label=random.choice(same_labels)
        # candidate1 = random.choice(
        #     [t for t in candidate_subtree1 if t.label() == select_label])
        # candidate2 = random.choice(
        #     [t for t in candidate_subtree2 if t.label() == select_label])
        candidate_subtree1 = list(t1.subtrees())
        candidate_subtree2 = list(t2.subtrees())
        candidate1 = random.choice(
            [t for t in candidate_subtree1])
        candidate2 = random.choice(
            [t for t in candidate_subtree2])
        exchanged_span = ' '.join(candidate1.leaves())
        exchanging_span = ' '.join(candidate2.leaves())
        original_sentence = ' '.join(t1.leaves())
        new_sentence = original_sentence.replace(exchanged_span, exchanging_span)
        debug=0
        if args.debug:
            print('check6')
            print(new_sentence)
            debug=1
        modified_sentence=modify(new_sentence,debug)
        new_label=c2a(modified_sentence,debug)
        if args.showinfo:
            print('cand1:', ' '.join(candidate1.leaves()),
                'cand2:', ' '.join(candidate2.leaves()))
            # print([' '.join(c.leaves()) for c in cand1])
            # print([' '.join(c.leaves()) for c in cand2])
            print('src1:', parsing1)
            print('src2:', parsing2)
            print('new:',new_sentence)
        return modified_sentence,new_label
    except Exception as e:
        if args.debug:
            print('Error!!')
            print(e)
        return None
def subtree_exchange_single(args,parsing1,label1,parsing2,label2,lam1,lam2):
    """
    For a pair sentence, exchange subtree and return a label based on subtree length
     
    Find the candidate subtree, and extract correspoding span, and exchange span
    
    """
    if args.debug:
        print('check4')
    assert lam1>lam2
    t1=Tree.fromstring(parsing1)
    original_sentence=' '.join(t1.leaves())
    t1_len=len(t1.leaves())
    candidate_subtree1=list(t1.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))
    t2=Tree.fromstring(parsing2)
    candidate_subtree2=list(t2.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))
    if args.debug:
        print('check5')
    # print('subtree1:',len(candidate_subtree1),'\nsubtree2:',len(candidate_subtree2))
    if len(candidate_subtree1)==0 or len(candidate_subtree2)==0:
     
        return None
    if args.debug:
        print('check6')
    if args.phrase_label:
        if args.debug:
            print('phrase_label')
        tree_labels1=[tree.label() for tree in candidate_subtree1]
        tree_labels2=[tree.label() for tree in candidate_subtree2]
        same_labels=list(set(tree_labels1)&set(tree_labels2))
        if not same_labels:
            # print('无相同类型的子树')
            return None
        if args.phrase_length:
            if args.debug:
                print('phrase_lable_length')
            candidate=[(t1,t2) for t1 in candidate_subtree1 for t2 in candidate_subtree2  if len(t1.leaves())==len(t2.leaves()) and t1.label()==t2.label()]
            candidate1,candidate2= random.choice(candidate)
        else:
            if args.debug:
                print('phrase_lable')
            select_label=random.choice(same_labels)
            candidate1=random.choice([t for t in candidate_subtree1 if t.label()==select_label])
            candidate2=random.choice([t for t in candidate_subtree2 if t.label()==select_label])
    else:
        if args.debug:
            print('no phrase_label')
        if args.phrase_length:
            if args.debug:
                print('phrase_length')
            candidate=[(t1,t2) for t1 in candidate_subtree1 for t2 in candidate_subtree2  if len(t1.leaves())==len(t2.leaves())]
            candidate1,candidate2= random.choice(candidate)
        else:
            if args.debug:
                print('normal TreeMix')
            candidate1=random.choice(candidate_subtree1) 
            candidate2=random.choice(candidate_subtree2) 
        
    exchanged_span=' '.join(candidate1.leaves())
    exchanged_len=len(candidate1.leaves())
    exchanging_span=' '.join(candidate2.leaves())
    new_sentence=original_sentence.replace(exchanged_span,exchanging_span)
    # if args.mixup_cross:
    new_label=np.zeros(len(args.label_list))

    exchanging_len=len(candidate2.leaves())
    new_len=t1_len-exchanged_len+exchanging_len
    
    new_label[int(label2)]+=exchanging_len/new_len
    new_label[int(label1)]+=(new_len-exchanging_len)/new_len
   
    # else:
    #     new_label=label1
    if args.showinfo:
        # print('树1 {}'.format(t1))
        # print('树2 {}'.format(t2))
        print('-'*50)
        print('candidate1:{}'.format([' '.join(x.leaves()) for x in candidate_subtree1]))
        print('candidate2:{}'.format([' '.join(x.leaves()) for x in candidate_subtree2]))
        print('sentence1 ## {}  [{}]\nsentence2 ## {}  [{}]'.format(original_sentence,label1,' '.join(t2.leaves()),label2))
        print('{}  <=========== {}'.format(exchanged_span,exchanging_span))
        print('new sentence: ## {}'.format(new_sentence))
        print('new label:[{}]'.format(new_label))
    return new_sentence,new_label
def subtree_exchange_pair(args,parsing11,parsing12,label1,parsing21,parsing22,label2,lam1,lam2):
    """
    For a pair sentence, exchange subtree and return a label based on subtree length
     
    Find the candidate subtree, and extract correspoding span, and exchange span
    
    """
    assert lam1>lam2
    lam2=lam1-0.2
    t11=Tree.fromstring(parsing11)
    t12=Tree.fromstring(parsing12)
    original_sentence1=' '.join(t11.leaves())
    t11_len=len(t11.leaves())
    original_sentence2=' '.join(t12.leaves())
    t12_len=len(t12.leaves())
    candidate_subtree11=list(t11.subtrees(lambda t: lam1>len(t.leaves())/t11_len>lam2))
    candidate_subtree12=list(t12.subtrees(lambda t: lam1>len(t.leaves())/t12_len>lam2))
    t21=Tree.fromstring(parsing21)
    t22=Tree.fromstring(parsing22)
    t21_len=len(t21.leaves())
    t22_len=len(t22.leaves())
    candidate_subtree21=list(t21.subtrees(lambda t: lam1>len(t.leaves())/t11_len>lam2))
    candidate_subtree22=list(t22.subtrees(lambda t: lam1>len(t.leaves())/t12_len>lam2))
    if args.showinfo:
        print('\n')
        print('*'*50)
        print('t11_len:{}\tt12_len:{}\tt21_len:{}\tt22_len:{}\ncandidate_subtree11:{}\ncandidate_subtree12:{}\ncandidate_subtree21:{}\ncandidate_subtree21:{}'
            .format(t11_len,t12_len,t21_len,t22_len,candidate_subtree11,candidate_subtree12,candidate_subtree21,candidate_subtree22))

    # print('subtree1:',len(candidate_subtree1),'\nsubtree2:',len(candidate_subtree2))
    if len(candidate_subtree11)==0 or len(candidate_subtree12)==0 or len(candidate_subtree21)==0 or len(candidate_subtree22)==0:
        # print("this pair fail",len(candidate_subtree1),len(candidate_subtree2))
        return None
    
    if args.phrase_label:
        tree_labels11=[tree.label() for tree in candidate_subtree11]
        tree_labels12=[tree.label() for tree in candidate_subtree12]
        tree_labels21=[tree.label() for tree in candidate_subtree21]
        tree_labels22=[tree.label() for tree in candidate_subtree22]
        same_labels1=list(set(tree_labels11)&set(tree_labels21))
        same_labels2=list(set(tree_labels12)&set(tree_labels22))
        if not (same_labels1 and same_labels2) :
            # print('无相同类型的子树')
            return None
        select_label1=random.choice(same_labels1)
        select_label2=random.choice(same_labels2)
        displaced1=random.choice([t for t in candidate_subtree11 if t.label()==select_label1])
        displacing1=random.choice([t for t in candidate_subtree21 if t.label()==select_label1])
        displaced2=random.choice([t for t in candidate_subtree12 if t.label()==select_label2])
        displacing2=random.choice([t for t in candidate_subtree22 if t.label()==select_label2])
    else:
        displaced1=random.choice(candidate_subtree11) 
        displacing1=random.choice(candidate_subtree21) 
        displaced2=random.choice(candidate_subtree12) 
        displacing2=random.choice(candidate_subtree22) 
    
    
    displaced_span1=' '.join(displaced1.leaves())
    displaced_len1=len(displaced1.leaves())
    displacing_span1=' '.join(displacing1.leaves())
    new_sentence1=original_sentence1.replace(displaced_span1,displacing_span1)
    
    displaced_span2=' '.join(displaced2.leaves())
    displaced_len2=len(displaced2.leaves())
    displacing_span2=' '.join(displacing2.leaves())
    new_sentence2=original_sentence2.replace(displaced_span2,displacing_span2)
    
    # if args.mixup_cross:
    new_label=np.zeros(len(args.label_list))
    displacing_len1=len(displacing1.leaves())
    displacing_len2=len(displacing2.leaves())
    new_len=t11_len+t12_len-displaced_len1-displaced_len2+displacing_len1+displacing_len2
    displacing_len=displacing_len1+displacing_len2
    new_label[int(label2)]+=displacing_len/new_len
    new_label[int(label1)]+=(new_len-displacing_len)/new_len
  
  
    if args.showinfo:
        print('Before\nsentence1:{}\nsentence2:{}\nlabel1:{}\tlabel2:{}'.format(original_sentence1,original_sentence2,label1,label2))
        print('replaced1:{} replacing1:{}\nreplaced2:{} replacing2:{}'.format(displaced_span1,displacing_span1,displaced_span2,displacing2))
        print('After\nsentence1:{}\nsentence2:{}\nnew_label:{}'.format(new_sentence1,new_sentence2,new_label))
        print('*'*50)

    # print('被替换的span:{}\n替换的span:{}'.format(exchanged_span,exchanging_span))
    return new_sentence1,new_sentence2,new_label
def augmentation(args,data,seed,dataset,aug_times,lam1=0.1,lam2=0.3):
    """
    generate aug_num augmentation dataset 
    input:
        dataset --- pd.dataframe
    output:
        aug_dataset --- pd.dataframe
    """
    generated_list=[]
    # print('check2')
    if args.debug:
        print('check3')
    shuffled_dataset=dataset.shuffle()
    success=0
    total=0
    with tqdm(total=int(aug_times)*len(dataset)) as bar:
        while success < int(aug_times)*len(dataset):
            # for idx in range(len(dataset)):
            idx = total % len(dataset)
            if args.fraction:
                bar.set_description('| Dataset:{:<5} | seed:{} | times:{} | fraction:{} |'.format(data,seed,aug_times,args.fraction))
            else:
                bar.set_description('| Dataset:{:<5} | seed:{} | times:{} | '.format(data,seed,aug_times))       
                
            if args.data_type=='single_cls':    
                if args.debug:
                    print('check4')
                if 'None' not in [dataset[idx]['parsing1'], shuffled_dataset[idx]['parsing1']]:
                    aug_sample=subtree_exchange_single(
                        args,dataset[idx]['parsing1'],dataset[idx][args.label_name],
                        shuffled_dataset[idx]['parsing1'],shuffled_dataset[idx][args.label_name],
                        lam1,lam2)
                else:
                    continue
                
            elif args.data_type=='pair_cls':
                # print('check4:pair')
                if args.debug:
                    print('check4')
                if 'None' not in [dataset[idx]['parsing1'], dataset[idx]['parsing2'], dataset[idx][args.label_name],
                                shuffled_dataset[idx]['parsing1'], shuffled_dataset[idx]['parsing2']]:
                    aug_sample=subtree_exchange_pair(
                        args,dataset[idx]['parsing1'],dataset[idx]['parsing2'],dataset[idx][args.label_name],
                        shuffled_dataset[idx]['parsing1'],shuffled_dataset[idx]['parsing2'],shuffled_dataset[idx][args.label_name],
                        lam1,lam2)
                else:
                    continue
                
            elif args.data_type=='semantic_parsing':  
                if args.debug:
                    print('check4')  
                aug_sample=subtree_exchange_scan(
                    args,dataset[idx]['parsing1'],
                    shuffled_dataset[idx]['parsing1'])
                
                if args.debug:
                    print('ok')
                    print('got one aug_sample : {}'.format(aug_sample))
            if  aug_sample: 
                bar.update(1)
                success+=1
                generated_list.append(aug_sample)
            else:
                if args.debug:
                    print('fail this time')
            total+=1
    #De-duplication
    # generated_list=list(set(generated_list))
    return generated_list
def parse_argument():
    parser=argparse.ArgumentParser() 
    parser.add_argument('--lam1',type=float,default=0.3)
    parser.add_argument('--lam2',type=float,default=0.1)
    parser.add_argument('--times',default=[2,5],nargs='+',help='augmentation times list')
    parser.add_argument('--min_token',type=int,default=0,help='minimum token numbers of augmentation samples')
    parser.add_argument('--label_name',type=str,default='label')
    parser.add_argument('--phrase_label',action='store_true',help='subtree lable must be same if set')
    parser.add_argument('--phrase_length',action='store_true',help='subtree phrase must be same length if set')
    # parser.add_argument('--data_type',type=str,required=True,help='This is a single classification task or pair sentences classification task')
    parser.add_argument('--seeds',default=[0,1,2,3,4],nargs='+',help='seed list')
    parser.add_argument('--showinfo',action='store_true')
    parser.add_argument('--mixup_cross',action='store_false',help="NO mix across different classes if set")
    parser.add_argument('--low_resource',action='store_true',help="create low source raw and aug datasets if set")
    parser.add_argument('--debug',action='store_true',help="display debug information")
    parser.add_argument('--data',nargs='+',required=True,help='data list')
    parser.add_argument('--proc',type=int,help='processing number for multiprocessing')
    args=parser.parse_args()
    if not args.proc:
        args.proc=cpu_count()
    return args
def create_aug_data(args,dataset,data,seed,times,test_dataset=None):

    if args.phrase_label and not args.phrase_length:
        prefix_save_path=os.path.join(args.output_dir,'samephraselabel_times{}_min{}_seed{}_{}_{}'.format(times,args.min_token,seed,args.lam1,args.lam2))
    elif args.phrase_length and not args.phrase_label:
        prefix_save_path=os.path.join(args.output_dir,'samephraselength_times{}_min{}_seed{}_{}_{}'.format(times,args.min_token,seed,args.lam1,args.lam2))
    elif args.phrase_length and args.phrase_label:
        prefix_save_path=os.path.join(args.output_dir,'samephraselabel_length_times{}_min{}_seed{}_{}_{}'.format(times,args.min_token,seed,args.lam1,args.lam2))
    elif not args.mixup_cross:
        prefix_save_path=os.path.join(args.output_dir,'sameclass_times{}_min{}_seed{}_{}_{}'.format(times,args.min_token,seed,args.lam1,args.lam2))
    elif args.data_type == 'semantic_parsing':
        prefix_save_path = os.path.join(args.output_dir, 'scan_times{}_seed{}'.format(
            times, seed))
    else:   
        prefix_save_path=os.path.join(args.output_dir,'times{}_min{}_seed{}_{}_{}'.format(times,args.min_token,seed,args.lam1,args.lam2))
    if args.debug:
        print('check1')
    if not [file_name for file_name in os.listdir(args.output_dir) if file_name.startswith(prefix_save_path)]:
        if args.min_token:
            dataset=dataset.filter(lambda sample: len(sample[tasksettings.task_to_keys[data][0]].split(' '))>args.min_token)
            if tasksettings.task_to_keys[data][1]:
                dataset=dataset.filter(lambda sample: len(sample[tasksettings.task_to_keys[data][1]].split(' '))>args.min_token)
        if args.data_type=='single_cls':
            if args.debug:
                print('check2')
            if args.mixup_cross:
                new_pd=pd.DataFrame(augmentation(args,data,seed,dataset,times,args.lam1,args.lam2),columns=[tasksettings.task_to_keys[data][0],args.label_name])
            else:
                if args.debug:
                    print('label_list',args.label_list)
                new_pd=None
                for i in args.label_list:
                    samples=dataset.filter(lambda sample:sample[args.label_name]==i)
                    dataframe=pd.DataFrame(augmentation(args,data,seed,samples,times,args.lam1,args.lam2),columns=[tasksettings.task_to_keys[data][0],args.label_name])
                    new_pd=pd.concat([new_pd,dataframe],axis=0)
        elif args.data_type=='pair_cls':
            if args.debug:
                print('check2')
            if args.mixup_cross:
                # print('check1')
                # print(args, seed, dataset, times,tasksettings.task_to_keys[data][0], tasksettings.task_to_keys[data][1], args.label_name)
                new_pd=pd.DataFrame(augmentation(args,data,seed,dataset,times,args.lam1,args.lam2),columns=[tasksettings.task_to_keys[data][0],tasksettings.task_to_keys[data][1],args.label_name])
            else:
                new_pd=None
                if args.debug:
                    print('label_list',args.label_list)
                for i in args.label_list:
                    samples=dataset.filter(lambda sample:sample[args.label_name]==i)
                    dataframe=pd.DataFrame(augmentation(args,data,seed,samples,times,args.lam1,args.lam2),columns=[tasksettings.task_to_keys[data][0],tasksettings.task_to_keys[data][1],args.label_name])
                    new_pd=pd.concat([new_pd,dataframe],axis=0)
        elif args.data_type=='semantic_parsing':
            if args.debug:
                print('check2')
            new_pd=pd.DataFrame(augmentation(args,data,seed,dataset,times),columns=[tasksettings.task_to_keys[data][0],args.label_name])
            
        
        new_pd=new_pd.sample(frac=1)
        
        
        
        if args.data_type == 'semantic_parsing':
            
            train_pd=pd.read_csv('DATA/ADDPRIM_JUMP/data/train.csv')
            frames = [train_pd,new_pd]
            aug_dataset=pd.concat(frames,ignore_index=True)
        else:
            aug_dataset = Dataset.from_pandas(new_pd)
            aug_dataset = aug_dataset.remove_columns("__index_level_0__")
            
        if args.phrase_label:
            save_path = os.path.join(args.output_dir, 'samephraselabel_times{}_min{}_seed{}_{}_{}_{}k'.format(
                times, args.min_token, seed, args.lam1, args.lam2, round(len(new_pd)//1000,-1)))
        elif args.phrase_length and not args.phrase_label:
            save_path=os.path.join(args.output_dir,'samephraselength_times{}_min{}_seed{}_{}_{}_{}k'.format(times,args.min_token,seed,args.lam1,args.lam2,round(len(new_pd)//1000,-1)))
        elif args.phrase_length and args.phrase_label:
            save_path=os.path.join(args.output_dir,'samephraselabel_length_times{}_min{}_seed{}_{}_{}_{}k'.format(times,args.min_token,seed,args.lam1,args.lam2,round(len(new_pd)//1000,-1)))
        elif not args.mixup_cross:
            save_path=os.path.join(args.output_dir,'sameclass_times{}_min{}_seed{}_{}_{}_{}k'.format(times,args.min_token,seed,args.lam1,args.lam2,round(len(new_pd)//1000,-1)))
        elif args.data_type=='semantic_parsing':
            save_path_train = os.path.join(prefix_save_path, 'train.csv')
            save_path_test = os.path.join(prefix_save_path, 'test.csv')
        else:   
            save_path=os.path.join(args.output_dir,'times{}_min{}_seed{}_{}_{}_{}k'.format(times,args.min_token,seed,args.lam1,args.lam2,round(len(new_pd)//1000,-1)))
        if args.data_type == 'semantic_parsing':
           

            if not os.path.exists(prefix_save_path):
                os.makedirs(prefix_save_path)

            aug_dataset.to_csv(save_path_train,index=0)
            test_dataset.to_csv(save_path_test,index=0)
        else:
            aug_dataset.save_to_disk(save_path)
    else:
        print('file {} already exsits!'.format(prefix_save_path))

    
def main():
    p=Pool(args.proc)
    for data in args.data:
        path_dir=os.path.join('DATA',data.upper())
        if data in tasksettings.pair_datasets:
            args.data_type='pair_cls'
        elif data in tasksettings.SCAN:
            args.label_name='actions'
            args.data_type='semantic_parsing'
            testset_path=os.path.join(path_dir,'data','test.csv')
        else:
            args.data_type='single_cls'
        if data=='trec':
            try:
                assert args.label_name in ['label-fine', 'label-coarse']
            except AssertionError:
                raise(AssertionError(
                    "If you want to train on trec dataset with augmentation, you have to name the label of split in ['label-fine', 'label-coarse']"))
            
            print(args.label_name,data)
            args.output_dir=os.path.join(path_dir,'generated/{}'.format(args.label_name))
        else:
            args.output_dir=os.path.join(path_dir,'generated')
        args.data_path=os.path.join(path_dir,'data','train_parsing.csv')
        
        dataset=load_dataset('csv',data_files=[args.data_path],split='train')
        if args.data_type=='semantic_parsing':
            testset=load_dataset('csv',data_files=[testset_path],split='train')
        if args.data_type in ['single_cls','pair_cls']:
            args.label_list=list(set(dataset[args.label_name])) #根据data做一个表查找所有的label 
        for seed in args.seeds:
            seed=int(seed)
            set_seed(seed)
            dataset=dataset.shuffle()
            if args.low_resource:
                for fraction in tasksettings.low_resource[data]:
                    args.fraction=fraction
                    train_dataset=dataset.select(random.sample(range(len(dataset)),int(fraction*len(dataset))))       
                    low_resource_dir=os.path.join(path_dir,'low_resource','low_resource_{}'.format(fraction),'seed_{}'.format(seed))
                    if not os.path.exists(low_resource_dir):
                        os.makedirs(low_resource_dir)
                    args.output_dir=low_resource_dir
                    train_path=os.path.join(args.output_dir,'partial_train')            
                    if not os.path.exists(train_path):
                        train_dataset.save_to_disk(train_path)
                    for times in args.times:
                        times=int(times)
                        p.apply_async(create_aug_data, args=(
                            args, train_dataset, data, seed, times))                     
            else:
                args.fraction=None
                for times in args.times:
                    times=int(times)
                    p.apply_async(create_aug_data, args=(
                        args, dataset, data, seed, times,testset))                   
        print('='*20,'Start generating augmentation datsets !',"="*20)
        # p.close()
        # p.join()
        
    p.close()
    p.join()
    print('='*20, 'Augmenatation done !', "="*20)
if __name__=='__main__':
    
    tasksettings=settings.TaskSettings()
    args=parse_argument()
    print(args)
    main()

from datasets import load_dataset
import numpy as np
import argparse
import os
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import time
import settings
from multiprocessing import  cpu_count
from pandarallel import pandarallel
def parse_argument():
    parser = argparse.ArgumentParser(description='download and parsing datasets')
    parser.add_argument('--data',nargs='+',required=True,help='data list')
    parser.add_argument('--corenlp_dir',type=str,default='/remote-home/lzhang/stanford-corenlp-full-2018-10-05/')
    parser.add_argument('--proc',type=int,help='multiprocessing num')
    args=parser.parse_args()
    return args


def parsing_stanfordnlp(raw_text):
    try:
        parsing = snlp.parse(raw_text)
        return parsing
    except Exception as e:
        return 'None'

def constituency_parsing(args):
    if not args.proc:
        args.proc = cpu_count()
    pandarallel.initialize(nb_workers=args.proc, progress_bar=True)
    for dataset in args.data:
        DATA_dir=os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")),'DATA')
        path_dir=os.path.join(DATA_dir,dataset.upper())
        output_path=os.path.join(path_dir,'data','train_parsing.csv')
        if os.path.exists(output_path):
            print('The data {} has already parsed!'.format(dataset.upper()))
            continue
        train=pd.read_csv(os.path.join(path_dir,'data','train.csv'),encoding="utf-8")
        for dataset in args.data:
            DATA_dir = os.path.join(os.path.abspath(
            os.path.join(os.getcwd(), "..")), 'DATA')
            path_dir = os.path.join(DATA_dir, dataset.upper())
            output_path = os.path.join(path_dir, 'data', 'train_parsing.csv')
            if os.path.exists(output_path):
                print('The data {} has already parsed!'.format(dataset.upper()))
                continue
            train = pd.read_csv(os.path.join(
                path_dir, 'data', 'train.csv'), encoding="utf-8")
            for i,text_name in enumerate(task_to_keys[dataset]):
                parsing_name = 'parsing{}'.format(i+1)
                train[parsing_name] = train[text_name].parallel_apply(
                    parsing_stanfordnlp)
        
            for i,text_name in enumerate(task_to_keys[dataset]): 
                parsing_name='parsing{}'.format(i+1)
                train=train.drop(train[train[parsing_name]=='None'].index)
            train.to_csv(output_path, index=0)
def download_data(args):  
    
    for dataset in args.data:
        DATA_dir=os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")),'DATA')
        path_dir=os.path.join(DATA_dir,dataset.upper())
        if dataset.upper() in os.listdir(DATA_dir):
            print('{} directory already exists !'.format(dataset.upper()))
            continue
        try:
            if dataset ==['addprim_jump','addprim_turn_left','simple']:
                downloaded_data_list = [load_dataset('scan', dataset)]
            if dataset in ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
                downloaded_data_list=[load_dataset('glue',dataset)]
            elif dataset =='sst':
                downloaded_data_list = [load_dataset("sst", "default")]
            else:
                downloaded_data_list=[load_dataset(dataset)]
            
            if not os.path.exists(path_dir):
                if dataset=='trec':
                    os.makedirs(os.path.join(path_dir,'generated/fine'))
                    os.makedirs(os.path.join(path_dir,'generated/coarse'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-coarse/raw'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-coarse/aug'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-coarse/raw_aug'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-fine/raw'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-fine/aug'))
                    os.makedirs(os.path.join(
                        path_dir, 'runs/label-fine/raw_aug'))
                else:
                    os.makedirs(os.path.join(path_dir,'generated'))
                    os.makedirs(os.path.join(path_dir,'runs/raw'))
                    os.makedirs(os.path.join(path_dir,'runs/aug'))
                    os.makedirs(os.path.join(path_dir,'runs/raw_aug'))
                os.makedirs(os.path.join(path_dir,'logs'))
                os.makedirs(os.path.join(path_dir,'data'))
                for downloaded_data in downloaded_data_list:
                    for data_split in downloaded_data:
                        dataset_split=downloaded_data[data_split]
                        dataset_split.to_csv(os.path.join(path_dir,'data',data_split+'.csv'),index=0)
        except Exception as e:
            print('Downloading failed on {}, due to error {}'.format(dataset,e))
if __name__=='__main__':
    args = parse_argument()
    tasksettings=settings.TaskSettings()
    task_to_keys=tasksettings.task_to_keys
    print('='*20,'Start Downloading Datasets','='*20)
    download_data(args)
    print('='*20,'Start Parsing Datasets','='*20)
    snlp = StanfordCoreNLP(args.corenlp_dir)
    constituency_parsing(args)

from datasets import load_dataset
import pandas as np
import argparse
import os
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import time
import settings
from multiprocessing import cpu_count
from pandarallel import pandarallel


def parse_argument():
    parser = argparse.ArgumentParser(
        description='download and parsing datasets')
    parser.add_argument('--data', nargs='+', required=True, help='data list')
    parser.add_argument('--corenlp_dir', type=str,
                        default='/remote-home/lzhang/stanford-corenlp-full-2018-10-05/')
    parser.add_argument('--proc', type=int, help='multiprocessing num')
    args = parser.parse_args()
    return args

def parsing_using_stanfordnlp(raw_text):
    try:
        parsing= snlp.parse(raw_text)
        return parsing
    except Exception as e:
        return 'None'

        
def constituency_parsing(args):
    if not args.proc:
        args.proc = cpu_count()
    pandarallel.initialize(nb_workers=args.proc, progress_bar=True)
    for dataset in args.data:
        DATA_dir = os.path.join(os.path.abspath(
            os.path.join(os.getcwd(), "..")), 'DATA')
        path_dir = os.path.join(DATA_dir, dataset.upper())
        output_path = os.path.join(path_dir, 'data', 'ceshi_parsing.csv')
        if os.path.exists(output_path):
            print('The data {} has already parsed!'.format(dataset.upper()))
            continue
        train = pd.read_csv(os.path.join(
            path_dir, 'data', 'test.csv'), encoding="utf-8")
        del train['Unnamed: 0']
        for i,text_name in enumerate(task_to_keys[dataset]):
            parsing_name = 'parsing{}'.format(i+1)
            train[parsing_name] = train[text_name].parallel_apply(
                parsing_using_stanfordnlp)
            
        for i,text_name in enumerate(task_to_keys[dataset]): 
            parsing_name='parsing{}'.format(i+1)
            train=train.drop(train[train[parsing_name]=='none'].index)
        train.to_csv(output_path, index=0)


if __name__ == '__main__':
    args = parse_argument()
    tasksettings = settings.TaskSettings()
    task_to_keys = tasksettings.task_to_keys
    snlp = StanfordCoreNLP(args.corenlp_dir)
    constituency_parsing(args)

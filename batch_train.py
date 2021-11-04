import argparse
import os
from process_data.settings import TaskSettings
def parse_argument():
    parser = argparse.ArgumentParser(description='download and parsing datasets')
    parser.add_argument('--data',type=str,required=True,help='data list')
    parser.add_argument('--aug_dir',help='Augmentation file directory')
    parser.add_argument('--seeds',default=[0,1,2,3,4],nargs='+',help='seed list')
    parser.add_argument('--modes',nargs='+',required=True,help='seed list')
    parser.add_argument('--label_name',type=str,default='label')
    # parser.add_argument('--batch_size',default=128,type=int,help='train examples in each batch')
    # parser.add_argument('--aug_batch_size',default=128,type=int,help='train examples in each batch')
    parser.add_argument('--random_mix',type=str,choices=['zero_one','zero','one','all'],help="random mixup ")
    parser.add_argument('--prefix',type=str,help="only choosing the datasets with the prefix,for ablation study")
    parser.add_argument('--GPU',type=int,default=0,help="available GPU number")
    parser.add_argument('--low_resource', action='store_true',
                        help='whther to train low resource dataset')
    
    args=parser.parse_args()
    if args.data=='trec':
        try:
            assert args.label_name in ['label-fine','label-coarse']
        except AssertionError:
            raise( AssertionError("If you want to train on TREC dataset with augmentation, you have to name the label of split either 'label-fine' or 'label-coarse'"))
        args.aug_dir = os.path.join('DATA', args.data.upper(), 'generated',args.label_name)
    if args.aug_dir is None :
        args.aug_dir=os.path.join('DATA',args.data.upper(),'generated')
    
    if 'aug' in args.modes:
        try:
            assert [file for file in os.listdir(args.aug_dir) if 'times' in file]
        except AssertionError:
            raise( AssertionError( "{}".format('This directory has no augmentation file, please input correct aug_dir!') ) )
    if args.low_resource:
        try:
            args.low_resource = os.path.join('DATA', args.data.upper(),'low_resource')
            assert os.path.exists(args.low_resource)
        except AssertionError:
            raise( AssertionError("There is no any low resource datasets in this data"))
 
    return args
def batch_train(args):
    for seed in args.seeds:
        # for aug_file in os.listdir(args.aug_dir):
            for mode in args.modes:
                if mode=='raw':
                    # data_path=os.path.join(args.aug_dir,aug_file)
                    if args.random_mix:
                        os.system('CUDA_VISIBLE_DEVICES={} python run.py --label_name {} --mode {}   --seed {} --data {} --random_mix {}  --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '.format(args.GPU,args.label_name,mode,int(seed),args.data,args.random_mix,**settings[args.data]))
                    else:
                        os.system('CUDA_VISIBLE_DEVICES={} python run.py --label_name {} --mode {}  --seed {} --data {}   --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '.format(args.GPU,args.label_name,mode,int(seed),args.data,**settings[args.data]))
                else:
                    for aug_file in os.listdir(args.aug_dir):
                        if args.prefix:
                            # only train on file with prefix
                            if aug_file.startswith(args.prefix):
                                aug_file_path = os.path.join(
                                    args.aug_dir, aug_file)
                                assert os.path.exists(aug_file_path)
                                os.system('CUDA_VISIBLE_DEVICES={} python run.py --label_name {} --mode {} --seed {} --data {} --data_path {} --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '.format(
                                    args.GPU, args.label_name, mode, int(seed), args.data, aug_file_path, **settings[args.data]))
                        else:
                            # train on every file in dir
                            aug_file_path = os.path.join(
                                args.aug_dir, aug_file)
                            assert os.path.exists(aug_file_path)
                            os.system('CUDA_VISIBLE_DEVICES={} python run.py --label_name {} --mode {} --seed {} --data {} --data_path {} --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '.format(
                                args.GPU, args.label_name, mode, int(seed), args.data, aug_file_path, **settings[args.data]))
def low_resource_train(args):
    for partial_split in os.listdir(args.low_resource):
        partial_split_path=os.path.join(args.low_resource,partial_split)
        args.output_dir = os.path.join(
            args.low_resource_dir, partial_split)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for seed_num in os.listdir(partial_split_path):
            partial_split_seed_path=os.path.join(partial_split_path,seed_num)
            for mode in args.modes:
                if mode=='raw':  
                    if args.random_mix:
                        os.system('CUDA_VISIBLE_DEVICES={} python run.py --low_resource_dir {} --seed {} --output_dir {} --label_name {} --mode {}   --data {} --random_mix {}  --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '
                                  .format(args.GPU, partial_split_seed_path, int(seed_num.split('_')[1]), args.output_dir, args.label_name, mode,  args.data, args.random_mix, **settings[args.data]))
                    else:
                        os.system('CUDA_VISIBLE_DEVICES={} python run.py --low_resource_dir {} --seed {} --output_dir {} --label_name {} --mode {}  --data {}   --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '
                                  .format(args.GPU, partial_split_seed_path, int(seed_num.split('_')[1]),args.output_dir, args.label_name, mode,  args.data, **settings[args.data]))
                elif mode=='raw_aug':
                    for aug_file in [file for file in os.listdir(partial_split_seed_path) if file.startswith('times')]:
                        aug_file_path=os.path.join(partial_split_seed_path,aug_file)
                        assert os.path.exists(aug_file_path)
                        os.system('CUDA_VISIBLE_DEVICES={} python run.py --low_resource_dir {} --seed {} --output_dir {} --label_name {} --mode {}  --data {} --data_path {} --epoch {epoch} --batch_size {batch_size} --aug_batch_size {aug_batch_size} --val_steps {val_steps} --max_length {max_length} --augweight {augweight} '.format(
                            args.GPU, partial_split_seed_path, int(seed_num.split('_')[1]) , args.output_dir, args.label_name, mode, args.data, aug_file_path, **settings[args.data]))
if __name__=='__main__':
    args=parse_argument()
    tasksettings=TaskSettings()
    settings=tasksettings.train_settings
    if args.low_resource:
        args.low_resource_dir=os.path.join('DATA',args.data.upper(),'runs','low_resource')
        if not os.path.exists(args.low_resource_dir):
            os.makedirs(args.low_resource_dir)
        low_resource_train(args)
    else:
        batch_train(args)

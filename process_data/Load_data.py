from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer
import torch
import numpy as np
import os
from . import settings
class DATA_process(object):
    def __init__(self, args=None):
        if args:
            print('Initializing with args')
            self.data = args.data if args.data else None
            self.task = args.task if args.task else None
            self.tokenizer = BertTokenizer.from_pretrained(
                args.model, do_lower_case=True) if args.model else None
            self.tasksettings = settings.TaskSettings()
            self.max_length = args.max_length if args.max_length else None
            self.label_name = args.label_name if args.label_name else None
            self.batch_size = args.batch_size if args.batch_size else None
            self.aug_batch_size=args.aug_batch_size if args.aug_batch_size else None
            self.min_train_token = args.min_train_token if args.min_train_token else None
            self.max_train_token = args.max_train_token if args.max_train_token else None
            self.num_proc = args.num_proc if args.num_proc else None
            self.low_resource_dir = args.low_resource_dir if args.low_resource_dir else None
            self.data_path = args.data_path if args.data_path else None
            self.random_mix = args.random_mix if args.random_mix else None
      
    def validation_data(self):
        validation_set = self.validationset(
            data=self.data)
        print('='*20,'multiprocess processing test dataset','='*20)
        # Process dataset to make dataloader
        if self.task == 'single':
            validation_set = validation_set.map(
                self.encode, batched=True, num_proc=self.num_proc)
        else:
            validation_set = validation_set.map(
                self.encode_pair, batched=True, num_proc=self.num_proc)
        # validation_set = validation_set.map(lambda examples: {'labels': examples[args.label_name]}, batched=True)
        validation_set = validation_set.rename_column(
            self.label_name, "labels")
        validation_set.set_format(type='torch', columns=[
                                  'input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        val_dataloader = torch.utils.data.DataLoader(
            validation_set, batch_size=self.batch_size, shuffle=True)
        return val_dataloader
    def encode(self, examples):
        return self.tokenizer(examples[self.tasksettings.task_to_keys[self.data][0]], max_length=self.max_length, truncation=True, padding='max_length')
    def encode_pair(self, examples):
        return self.tokenizer(examples[self.tasksettings.task_to_keys[self.data][0]], examples[self.tasksettings.task_to_keys[self.data][1]], max_length=self.max_length, truncation=True, padding='max_length')

    def train_data(self, count_label=False):
        train_set, label_num = self.traindataset(
            data=self.data, low_resource_dir=self.low_resource_dir, label_num=count_label)
        print('='*20,'multiprocess processing train dataset','='*20)
        if self.task == 'single':
            train_set = train_set.map(
                self.encode, batched=True, num_proc=self.num_proc)
        else:
            train_set = train_set.map(
                self.encode_pair, batched=True, num_proc=self.num_proc)
        if self.random_mix:
            # sort the train dataset
            print('-'*20, 'random_mixup', '-'*20)
            train_set = train_set.map(
                lambda examples: {'token_num': np.sum(np.array(examples['attention_mask']))})
            train_set = train_set.sort('token_num', reverse=True)
        # train_set = train_set.map(lambda examples: {'labels': examples[args.label_name]}, batched=True)
        train_set = train_set.rename_column(self.label_name, "labels")
        if self.min_train_token:
            print(
                '-'*20, 'filter sample whose sentence shorter than {}'.format(self.min_train_token), '-'*20)
            train_set = train_set.filter(lambda example: sum(
                example['attention_mask']) > self.min_train_token+2)
        if self.max_train_token:
            print(
                '-'*20, 'filter sample whose sentence longer than {}'.format(self.max_train_token), '-'*20)
            train_set = train_set.filter(lambda example: sum(
                example['attention_mask']) < self.max_train_token+2)
        train_set.set_format(type='torch', columns=[
                             'input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        
        train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        if count_label:
            return train_dataloader, label_num
        else:
            return train_dataloader
    def augmentation_data(self):
        try:
            aug_dataset = load_dataset(
                'csv', data_files=[self.data_path])['train']
        except Exception as e:
            aug_dataset = load_from_disk(self.data_path)
        print('='*20, 'multiprocess processing aug dataset', '='*20)
        if self.task == 'single':
            aug_dataset = aug_dataset.map(
                self.encode, batched=True, num_proc=self.num_proc)
        else:
            aug_dataset = aug_dataset.map(
                self.encode_pair, batched=True, num_proc=self.num_proc)
        # if self.mix:
        #     # label has more than one dimension
        #     # aug_dataset = aug_dataset.map(lambda examples: {'labels':examples[self.label_name]},batched=True)
        # else:
        #     # aug_dataset = aug_dataset.map(lambda examples: {'labels':int(examples[self.label_name])})
        aug_dataset = aug_dataset.rename_column(self.label_name, 'labels')

        aug_dataset.set_format(type='torch', columns=[
                               'input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        aug_dataloader = torch.utils.data.DataLoader(
            aug_dataset, batch_size=self.aug_batch_size, shuffle=True)
        return aug_dataloader

    def validationset(self,data):
        if data in ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
            if data == 'mnli':
                validation_set = load_dataset(
                    'glue', data, split='validation_mismatched')
            else:
                validation_set = load_dataset('glue', data, split='validation')
            print('-'*20, 'Test on glue@{}'.format(data), '-'*20)
        elif data in ['imdb', 'ag_news', 'trec']:
            validation_set = load_dataset(data, split='test')
            print('-'*20, 'Test on {}'.format(data), '-'*20)
        elif data == 'sst':
            validation_set = load_dataset(data, 'default', split='test')
            validation_set = validation_set.map(lambda example: {'label': int(
                example['label']*10//2)}, remove_columns=['tokens', 'tree'], num_proc=4)
            print('-'*20, 'Test on {}'.format(data), '-'*20)
        else:
            validation_set = load_dataset(data, split='validation')
            print('-'*20, 'Test on {}'.format(data), '-'*20)
        
        return validation_set

    def traindataset(self, data, low_resource_dir=None, split='train', label_num=False):
        if low_resource_dir:
            train_set = load_from_disk(os.path.join(
                low_resource_dir, 'partial_train'))
        else:
            if data in ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
                train_set = load_dataset('glue', data, split=split)
            elif data == 'sst':
                train_set = load_dataset(data, 'default', split=split)
                train_set = train_set.map(lambda example: {'label': int(
                    example['label']*10//2)}, remove_columns=['tokens', 'tree'], num_proc=4)
            else:
                train_set = load_dataset(data, split=split)
        if label_num:
            return train_set, len(set(train_set[self.label_name]))
        else:
            return train_set
if __name__=="__main__":
    data_processor=DATA_process()
    valset=data_processor.validationset(data='ag_news')
    print(valset)

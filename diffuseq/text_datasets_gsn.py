# import blobfile as bf
import pdb

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
from diffuseq import MAX_UTTERANCE_NUM, MAX_UTTERANCE_TOKEN


def load_data_text(
        batch_size,
        seq_len,
        deterministic=False,
        data_args=None,
        model_emb=None,
        split='train',
        loaded_vocab=None,
        loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#' * 30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    if (training_data == None):
        print("--成功完成分叉")
        return None

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        # drop_last=True,
        shuffle=not deterministic,
        num_workers=0,
    )

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])

        input_split_ids = []
        for i in range(len(examples['history'])):
            x_split_encode = vocab_dict.encode_token(examples['history'][i])
            y_split_encode = input_id_y[i]
            x_split_encode.append(y_split_encode)
            input_split_ids.append(x_split_encode)
            # if(i == 0):
            #     print(x_split_encode )

        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y, 'input_split_ids': input_split_ids}

        return result_dict


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )              

    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src) > len(trg):
                    src.pop()
                elif len(src) < len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0] * (len(src) + 1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask


        for i in range(len(group_lst['input_split_ids'])):
            if(len(group_lst['input_split_ids'][i])>MAX_UTTERANCE_NUM):
                group_lst['input_split_ids'][i] = group_lst['input_split_ids'][i][:MAX_UTTERANCE_NUM]
            for j in range(len(group_lst['input_split_ids'][i])):
                if(len((group_lst['input_split_ids'][i][j]))>MAX_UTTERANCE_TOKEN):
                    group_lst['input_split_ids'][i][j]=group_lst['input_split_ids'][i][j][:MAX_UTTERANCE_TOKEN]

        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge_and_mask",
    )



    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        group_lst['input_split_ids'] = _collate_batch_helper_split(group_lst['input_split_ids'],
                                                                    vocab_dict.pad_token_id)

        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    raw_datasets = datasets.DatasetDict()
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets['train'] = lm_datasets
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src': [], 'trg': [], 'relation_at': [], 'history': [], 'tgt_idx':[]}

    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    if (split == 'train'):  # 首先自定义字表加载是在train， valid和test不重新制作，只是加载
        from spacy.lang.en import English
        from collections import Counter
        nlp = English()
        nlp_tokenizer = nlp.tokenizer

        word_lsts = []
        with open(path, 'r') as f_reader:
            for row in f_reader:
                src = json.loads(row, strict=False)['src'].strip()
                src = src.replace('\x01', ' ')
                trg = json.loads(row, strict=False)['trg'].strip()
                relation_at = json.loads(row, strict=False)['relation_at']
                relation_at = relation_at + (MAX_UTTERANCE_NUM - 1 - len(relation_at)) * [[-1, -1]]  # 将长度补齐,最多为8个句子，也就是7条边
                sentence_lst['src'].append(src)
                sentence_lst['trg'].append(trg)
                sentence_lst['relation_at'].append(relation_at)
                
                word_lst = [x.text for x in nlp_tokenizer(src + trg)]
                word_lsts.append(word_lst)

        if (data_args.vocab == 'to_design'):  # 只有是在自定义词表才会制作，如果是bert就不制作了
            from basic_utils import load_tokenizer
            counter = Counter()
            for input_ids in word_lsts:
                counter.update(input_ids)
            vocab_dict = []  # 使用列表，这里不需要值，只需要键，故用列表就行
            for k, v in counter.items():  # [ (key = word,value = times), ...;]
                if v > 10:
                    vocab_dict.append(k)  # 如果出现次数多余10次，则将其值放入表中
            path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
            print(f'save the vocab to {path_save_vocab}')

            with open(path_save_vocab, 'w') as f:
                for key in vocab_dict:  # 一个一个写入
                    f.write(key + '\n')
            return None
    else:
        with open(path, 'r') as f_reader:
            for row in f_reader:
                src = json.loads(row, strict=False)['src'].strip()
                history = src.split('\x01')
                src = src.replace('\x01', ' ')
                trg = json.loads(row, strict=False)['trg'].strip()
                tgt = json.loads(row, strict=False)['ans_idx']
                relation_at = json.loads(row, strict=False)['relation_at']
                relation_at = relation_at + (MAX_UTTERANCE_NUM - 1 - len(relation_at)) * [[-1, -1]]  # 将长度补齐,最多为8个句子，也就是7条边
                sentence_lst['src'].append(src)
                sentence_lst['trg'].append(trg)
                sentence_lst['relation_at'].append(relation_at)
                sentence_lst['tgt_idx'].append(tgt)
                sentence_lst['history'].append(history)

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2],
          sentence_lst['relation_at'][:2])

    vocab_dict = loaded_vocab


    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)

    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            # input_id_x = self.text_datasets['train'][idx]['input_id_x']

            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
            out_kwargs['input_split_ids'] = torch.tensor(self.text_datasets['train'][idx]['input_split_ids'])
            out_kwargs['relation_at'] = torch.tensor(self.text_datasets['train'][idx]['relation_at']).long()
            out_kwargs['tgt_idx'] = torch.tensor(self.text_datasets['train'][idx]['tgt_idx']).long()

            return arr, out_kwargs



def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def _collate_batch_helper_split(examples, pad_token_id,  return_mask=False):
    result_split = torch.full([len(examples), MAX_UTTERANCE_NUM, MAX_UTTERANCE_TOKEN], pad_token_id,
                              dtype=torch.int64).tolist()

    for i, sentence in enumerate(examples):
        for j, sub_sen in enumerate(sentence):
            sub_sen_token = min(len(sub_sen), MAX_UTTERANCE_TOKEN)
            result_split[i][j][:sub_sen_token] = sub_sen[:sub_sen_token]

    #print('result_split[0]', result_split[0])
    return result_split
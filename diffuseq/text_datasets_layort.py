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


def helper_tokenize_sample(sentence_lst, vocab_dict, seq_len):
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    # 传进来的sentence-lst有下面四个key
    # sentence_lst['src'].append(src)
    # sentence_lst['trg'].append(trg)
    # sentence_lst['index_lst'].append(index_lst)
    # sentence_lst['relation_at'].append(relation_at)

    # seq-len为下采样过程中，x与y拼接后的长度不能超过128个token
    def tokenize_function(examples):

        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])

        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict
    #  对src和trg分别进行tokenizer话，转化成id，分别存放在input_id_x，input_id_y中
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        # load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )                                
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    # 返回的值为字典，有'relation_at', 'index_lst', 'input_id_x', 'input_id_y'这4个key

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

            lst.append(src + [vocab_dict.sep_token_id] + trg + [vocab_dict.pad_token_id] * (seq_len-len(src)-len(trg)-1))
            mask.append([0] * (len(src) + 1) + [1]*(seq_len-len(src)-1))  # 这里从0 改成了 1
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge_and_mask",
    )

    # 返回的值为字典，有'relation_at','input_ids', 'index_lst', 'input_id_x', 'input_id_y','input_mask'这6个key

    def split_uttrance(group_lst):
        res = []
        # 将input_id_x分隔几个小句，将这一长句的分词后每个token对应的id，分配给每个子句并进行padding，
        # 保证每个长句分为8个子句，每个子句为30个token
        # 将上述结果赋给input_split_ids变量
        # 原始input_id_x+input_id_y的结果保存到

        print('len(group_lst[input_id_x]', len(group_lst['input_id_x']))
        print('(tokenized_datasets[input_id_x]',tokenized_datasets['input_id_x'][0])
        for i in range(len(group_lst['input_id_x'])):#300
            sentence = []
            middle =[]
            for j in range(len(group_lst['index_lst'][i])-1):
                middle.append((group_lst['index_lst'][i][j],group_lst['index_lst'][i][j+1]))

            for start, end in middle:
                if  end == -1:
                    print('start,end come here', start, end)
                    sentence.append([0] * MAX_UTTERANCE_TOKEN)
                elif end-start <= MAX_UTTERANCE_TOKEN:
                    remain0 = MAX_UTTERANCE_TOKEN - (end-start)
                    sentence.append(group_lst['input_id_x'][i][start:end]+[0]*remain0)
                elif end-start > MAX_UTTERANCE_TOKEN:
                    sentence.append(group_lst['input_id_x'][i][start:start+MAX_UTTERANCE_TOKEN])
            if (i == 32):
                print('(tokenized_datasets[input_id_x]',group_lst['input_id_x'][i])
                print("middle",middle)
                print('sentence', sentence)
                exit(0)
            res.append(sentence)
        
        group_lst['input_split_ids'] = res

        return group_lst
    
    def split_uttrance2(group_lst):
        #索引必须转换成long
        group_lst['index_lst'] = torch.tensor(group_lst['index_lst']).long()
        group_lst['tgt_idx'] = torch.tensor(group_lst['tgt_idx']).long()
        res= torch.full( ( len(group_lst['input_id_x']) ,MAX_UTTERANCE_NUM, MAX_UTTERANCE_TOKEN), -1)
        middle =torch.zeros(( len(group_lst['input_id_x']),MAX_UTTERANCE_NUM-1,2),dtype = torch.long )
        print(group_lst['index_lst'].shape)
        #print(type(group_lst['index_lst']),group_lst['index_lst'])
        middle[:,:,0]   = group_lst['index_lst'][:,:-1]  #  0, 16, 87, 99, 198,-1
        middle[:,:,1] = group_lst['index_lst'][:,1:]#  16, 87, 99, 198 ,-1,-1
        
        for i in range(len(group_lst['input_id_x'])):
            for j in  range(len(group_lst['index_lst'][i])):
                if( middle[i][j][1] != -1 ):
                    print(group_lst['input_id_x'][i])
                    res[i,j,0:middle[i,j,1]-middle[i,j,0]] = group_lst['input_id_x'][i][ middle[i,j,0]:middle[i,j,1] ]
                    print(group_lst['input_id_x'][i][j][middle[i,j][0]:middle[i,j][1]])
                    exit(0)
                else:
                    break
        group_lst['input_split_ids'] = res
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        split_uttrance2,
        batched=True,
        num_proc=1,
        desc=f"split_uttrance",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = tokenized_datasets
    #返回值为 'relation_at', 'index_lst', 'input_id_x', 'input_id_y', 'input_ids', 'input_mask','input_split_ids'这7个字典
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets['train'] = tokenized_datasets
    return raw_datasets


def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        print('### before tokenized ', examples['src'][0])
        print('### before tokenized**len ', len(examples['src'][0].split(' ')))
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])

        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        # load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets***len ', len(tokenized_datasets['input_id_x']))
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print('### tokenized_datasets...example***len', len(tokenized_datasets['input_id_x'][0]))
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
            mask.append([1] * (len(src) + 1))  # 这里从 0 改成了 1 
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )

    # def pad_function(group_lst):
    #     max_length = seq_len
    #     group_lst['input_ids'] = _collate_batch_helper_sample(group_lst['input_ids'], vocab_dict.pad_token_id,max_length)
    #     group_lst['input_mask'] = _collate_batch_helper_sample(group_lst['input_mask'], 1, max_length)
    #     return group_lst
    #
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #
    # lm_datasets = tokenized_datasets.map(
    #     pad_function,
    #     batched=True,
    #     num_proc=1,
    #     desc=f"padding",
    # )
    #
    # print(lm_datasets, 'padded dataset')
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    # print('### padded_datasets...grouptext', lm_datasets['input_id_x'][0])
    # print('### padded_datasets...grouptext***len', len(lm_datasets['input_id_x']))
    #
    # raw_datasets = datasets.DatasetDict()
    # raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src': [], 'trg': [], 'relation_at': [], 'index_lst': []}

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
                index_lst = _get_utterance_length(src)
                src = src.replace('\x01', ' ')
                trg = json.loads(row, strict=False)['trg'].strip()
                relation_at = json.loads(row, strict=False)['relation_at']
                relation_at = relation_at + (MAX_UTTERANCE_NUM - 1 - len(relation_at)) * [
                    [-1, -1]]  # 将长度补齐,最多为8个句子，也就是7条边

                index_lst = index_lst + (MAX_UTTERANCE_NUM - 1 - len(index_lst)) * [
                    -1]  # 将长度补齐,最多为8个句子，也就是7个分隔，trg天然分开，所以再减一个分隔量

                sentence_lst['src'].append(src)
                sentence_lst['trg'].append(trg)
                sentence_lst['index_lst'].append(index_lst)
                sentence_lst['relation_at'].append(relation_at)
                word_lst = [x.text for x in nlp_tokenizer(src + trg)]
                word_lsts.append(word_lst)

        if (data_args.vocab == 'to_design'):  # 只有是在自定义词表才会制作，如果是bert就不制作了
            from basic_utils import load_tokenizer

            counter = Counter()
            for input_ids in word_lsts:
                counter.update(input_ids)

            # vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
            vocab_dict = []  # 使用列表，这里不需要值，只需要键，故用列表就行
            for k, v in counter.items():  # [ (key = word,value = times), ...;]
                if v > 10:
                    # vocab_dict[k] = len(vocab_dict)
                    vocab_dict.append(k)  # 如果出现次数多余10次，则将其值放入表中
            path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
            print(f'save the vocab to {path_save_vocab}')
            with open(path_save_vocab, 'w') as f:
                # json.dump(vocab_dict,f)
                for key in vocab_dict:  # 一个一个写入
                    f.write(key + '\n')
            return None
    else:
        with open(path, 'r') as f_reader:
            for row in f_reader:
                src = json.loads(row, strict=False)['src'].strip()
                index_lst = _get_utterance_length(src)
                src = src.replace('\x01', ' ')
                trg = json.loads(row, strict=False)['trg'].strip()
                #目标句子
                tgt = json.loads(row, strict=False)['ans_idx'].strip()

                relation_at = json.loads(row, strict=False)['relation_at']
                relation_at = relation_at + (MAX_UTTERANCE_NUM - 1 - len(relation_at)) * [
                    [-1, -1]]  # 将长度补齐,最多为8个句子，也就是7条边

                index_lst = index_lst + (MAX_UTTERANCE_NUM - len(index_lst)) * [
                    -1]  # 将长度补齐,最多为8个句子，也就是7个分隔\x01，加最后结尾，trg天然分开，
                sentence_lst['src'].append(src)
                sentence_lst['trg'].append(trg)
                sentence_lst['index_lst'].append(index_lst)
                sentence_lst['relation_at'].append(relation_at)
                sentence_lst['tgt_idx'].append(tgt)


    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2],
          sentence_lst['relation_at'][:2], sentence_lst['index_lst'][:2])

    # print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2], sentence_lst['history'][:2], sentence_lst['relation_at'][:2], sentence_lst['src_len'][:2])

    # get tokenizer.
    vocab_dict = loaded_vocab

    # src是原始最长的token，trg是回复的token，index_lst
    train_dataset = helper_tokenize_sample(sentence_lst, vocab_dict, seq_len)
    # train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb
        with torch.no_grad():
            text_datasets['train']['hidden_state'] = self.model_emb(torch.tensor(text_datasets['train']['input_id_x']))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            # input_id_x = self.text_datasets['train'][idx]['input_id_x']

            input_ids = self.text_datasets['train'][idx]['input_id_x']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_id_x'])
            
            #out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
            out_kwargs['input_split_hidden'] = self.model_emb(torch.tensor(self.text_datasets['train'][idx]['input_split_ids']))
            out_kwargs['relation_at'] = torch.tensor(self.text_datasets['train'][idx]['relation_at']).long()
            out_kwargs['tgt_idx'] = torch.tensor(self.text_datasets['train'][idx]['tgt_idx']).long()
            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    print('pad_token_id', pad_token_id)
    print('len(examples)', len(examples))
    print('max_length', max_length)
    print('type(examples)', type(examples))
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):

        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        if (i == 0):
            print('example', example)
            print('result[0]', result[i])
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    print('result', result[0])
    print('len(result[0])', len(result[0]))
    return result


# 用来得到不同句子的位置，用来分割
# 这里的是对于句子tokenizer化后的 每个字句结尾所在总句的位置信息

def _get_utterance_length(src):
    from spacy.lang.en import English
    nlp = English()
    nlp_tokenizer = nlp.tokenizer
    #先分词，把每个子句的最后一个token对应的len(token(sub))即为(下标+1)存放，为了后续生成[start：end）的token
    length_lst = [0]
    sub_src = src.split("\x01")
    for sub in sub_src:
        length_lst.append(len(nlp_tokenizer(sub)) + length_lst[-1])
    return length_lst

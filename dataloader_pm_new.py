'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
from gensim.parsing.preprocessing import remove_stopwords
import torch.nn.functional as F
from transformers import RobertaTokenizer, BertTokenizer

np.random.seed(1234)
torch.manual_seed(1234)


def tokenize_separate(data, tokenizer, speaker, model_type='roberta-large'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    for key in keys:
        dial = data[key]
        # for idx, _ in enumerate(dial):
        #     dial[idx] = re.sub(r'\x92', '', dial[idx])
        #     dial[idx] = re.sub(r'\x91', '', dial[idx])
        speaker_ = speaker[key]
        for idx, (sp, d) in enumerate(zip(speaker_, dial)):
            sp_index = torch.where(sp == 1)[0].item()
            dial[idx] = tokenizer.additional_special_tokens[sp_index] + ' ' + dial[idx]
        res = tokenizer(dial, padding='longest', return_tensors='pt')
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        if model_type in ['roberta', 'roberta-large']:
            token_types[key] = []
        elif model_type in ['bert', 'bert-large']:
            token_types[key] = res['token_type_ids']

    return input_ids, masks, token_types


def tokenize(data, tokenizer, model_type='roberta-large'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    count = 0
    for key in keys:
        dial = data[key]

        res = tokenizer(dial, padding='longest', return_tensors='pt')
        if res['input_ids'].size(1) > 512:
            pun = '!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~'
            dial = re.sub(r'[{}]+'.format(pun), '', dial)
            dial = remove_stopwords(dial)
            res = tokenizer(dial, padding='longest', return_tensors='pt')
            if res['input_ids'].size(1) > 512:
                dial = re.sub(r'http\S+', 'link', dial)
                res = tokenizer(dial, padding='longest', return_tensors='pt')
                # print(res['input_ids'].size(1))
            count += 1
            # print(res['input_ids'].size(1))
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        if model_type in ['roberta', 'roberta-large']:
            token_types[key] = []
        elif model_type in ['bert', 'bert-large']:
            token_types[key] = res['token_type_ids']

    # print('count->>>>', count)

    return input_ids, masks, token_types


def concate_sen(sentences, speakers, tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''
    max_sp = 0
    for key in keys:
        dialog = sentences[key]
        speaker = speakers[key]
        for idx, (sp, dial) in enumerate(zip(speaker, dialog)):
            sp_index = torch.where(sp==1)[0].item()
            tokenizer.additional_special_tokens[sp_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index] + ' ' + dial
            else:
                concatenated += tokenizer.sep_token + ' ' + tokenizer.additional_special_tokens[sp_index] + ' ' + dial
        concatenated_sentences[key] = concatenated
        concatenated = ''

    return concatenated_sentences


def tokenize_pre(data, tokenizer, speaker):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    for key in keys:
        dial = data[key]
        # for idx, _ in enumerate(dial):
        #     dial[idx] = re.sub(r'\x92', '', dial[idx])
        #     dial[idx] = re.sub(r'\x91', '', dial[idx])
        speaker_ = speaker[key]
        for idx, (sp, d) in enumerate(zip(speaker_, dial)):
            sp_index = torch.where(sp == 1)[0].item()
            dial[idx] = tokenizer.additional_special_tokens[sp_index] + ' ' + dial[idx]
        res = tokenizer(dial, padding='longest', return_tensors='pt')
        if res['input_ids'].size(1) > 512:
            pun = '!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~'
            dial = [re.sub(r'[{}]+'.format(pun), '', i) for i in dial]
            dial = [remove_stopwords(i) for i in dial]
            # dial = re.sub(r'[{}]+'.format(pun), '', dial)
            # dial = remove_stopwords(dial)
            res = tokenizer(dial, padding='longest', return_tensors='pt')
            if res['input_ids'].size(1) > 512:
                # dial = re.sub(r'http\S+', 'link', dial)
                dial = [re.sub(r'http\S+', 'link', i) for i in dial]
                res = tokenizer(dial, padding='longest', return_tensors='pt')

        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        token_types[key] = []

    return input_ids, masks, token_types


def concate_sen_pre(sentences, speakers, tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''
    concatenated_pre = []
    # max_sp = 0
    for key in keys:
        dialog = sentences[key]
        speaker = speakers[key]
        for idx, (sp, dial) in enumerate(zip(speaker, dialog)):
            sp_index = torch.where(sp==1)[0].item()
            tokenizer.additional_special_tokens[sp_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index] + ' ' + dial
                concatenated_pre.append(concatenated)
            else:
                concatenated += tokenizer.sep_token + ' ' + tokenizer.additional_special_tokens[sp_index] + ' ' + dial
                concatenated_pre.append(concatenated)
        concatenated_sentences[key] = concatenated_pre
        concatenated = ''
        concatenated_pre = []

    return concatenated_sentences


class DialogDataset(Dataset):

    def __init__(self, dataset=None, model_type='roberta-large', single=False, multi_par=True, n_rels=16):
        if isinstance(dataset, list):

            self.data = {}

            for idx, dialog_ in enumerate(dataset):
                if len(dialog_["relations"]) == 0:
                    continue
                self.data[idx] = dialog_
            del dataset
        else:
            self.data = {}
            for idx, (ids, dialog_) in enumerate(dataset.items()):
                if len(dialog_["relations"]) == 0 or len(dialog_["edus"]) == 1:
                    continue
                self.data[idx] = dialog_
            del dataset

        # sort the dialogues according to the length of the dialog
        self.sorted_data = sorted(self.data.items(), key=lambda x: len(x[1]['edus']))
        self.keys = [i[0] for i in self.sorted_data]
        self.index_map = {idx: i[0] for idx, i in enumerate(self.sorted_data)}
        self.len = len(self.keys)
        self.length = {}

        self.input = {}
        self.label = {}
        self.type_label = {}
        self.type_label_train = {}
        self.mask = {}
        self.speaker = {}
        self.speaker_matrix = {}
        self.turn = {}
        self.turn_matrix = {}
        self.src_sp = {}
        self.dst_sp = {}
        self.src_turn = {}
        self.dst_turn = {}

        self.label_td = {}
        self.type_label_td = {}
        if model_type in ['roberta', 'roberta-large']:
            if model_type == 'roberta-large':
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            elif model_type == 'roberta':
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type in ['bert', 'bert-large']:
            if model_type == 'bert-large':
                self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking') # TODO
                # self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            elif model_type == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.slide_win = max_edu_dist
        self.single = single

        for id in tqdm(self.keys):
            inputs = []
            # label = []
            speaker = []
            turn = []
            self.src_sp[id] = []
            self.dst_sp[id] = []
            self.src_turn[id] = []
            self.dst_turn[id] = []

            dialog = self.data[id]
            dial = dialog['edus']
            relations = dialog['relations']

            # type_la_train = torch.zeros(len(dial), len(dial) + 1).long() - 1
            if multi_par:
                la_new = torch.zeros(len(dial), len(dial)).long()
            else:
                la_new = torch.zeros(len(dial)).long()
                type_la = [-1 for _ in range(len(dial))]

            type_la_td = torch.zeros(len(dial), len(dial), n_rels).long()

            if len(relations) == 0:
                # label.append([0, 0, 0])
                continue
            else:
                for la in relations:
                    if la['x'] > la['y']:
                        continue
                    # type_la_train[la['y']-1][la['x']] = la['type']
                    if multi_par:
                        la_new[la['y']][la['x']] = 1
                    else:
                        type_la[la['y']] = la['type']
                        la_new[la['y']] = la['x']
                    type_la_td[la['x']][la['y']][la['type']] = 1
            if multi_par:
                la_td = la_new.T
                type_la = type_la_td.transpose(0, 1)
            else:
                la_td = F.one_hot(la_new, num_classes=la_new.size(0)).T

            self.cnt_speaker = {}
            count_speaker = 0
            # speaker_matrix = np.zeros([len(dial), len(dial)])
            # turn_matrix = np.zeros([len(dial), len(dial)])
            for idx_i, utt in enumerate(dial):
                utt_ = utt['text_raw']
                inputs.append(utt_)
                speaker.append(utt['speaker'])
                if utt['speaker'] not in self.cnt_speaker:
                    self.cnt_speaker[utt['speaker']] = count_speaker
                    count_speaker += 1
                turn.append(utt['turn'])

                if idx_i <= len(dial) - 1:
                    for idx_j in range(0, len(dial)):
                            if dial[idx_i]['speaker'] == dial[idx_j]['speaker']:
                                # speaker_matrix[idx_i][idx_j] = 1
                                self.src_sp[id].append(idx_i)
                                self.dst_sp[id].append(idx_j)
                            if dial[idx_i]['turn'] == dial[idx_j]['turn']:
                                # turn_matrix[idx_i][idx_j] = 1
                                self.src_turn[id].append(idx_i)
                                self.dst_turn[id].append(idx_j)


            self.input[id] = inputs
            # self.speaker_matrix[id] = torch.Tensor(speaker_matrix)
            # self.src_sp[id] = torch.LongTensor(self.src_sp[id])
            # self.dst_sp[id] = torch.LongTensor(self.dst_sp[id])
            self.label[id] = la_new
            self.type_label[id] = torch.LongTensor(type_la)
            # self.type_label_train[id] = type_la_train
            speaker_ = [F.one_hot(torch.tensor(self.cnt_speaker[s]), num_classes=count_speaker) for idx, s in enumerate(speaker)]
            self.speaker[id] = torch.stack(speaker_)  # torch.from_numpy(np.array(speaker))
            self.turn[id] = torch.from_numpy(np.array(turn))
            # self.turn_matrix[id] =torch.Tensor(turn_matrix)
            # self.src_turn[id] = torch.LongTensor(self.src_turn[id])
            # self.dst_turn[id] = torch.LongTensor(self.dst_turn[id])

            self.label_td[id] = la_td
            self.type_label_td[id] = type_la_td

        special_tokens_dict = {
            'additional_special_tokens': ['</s0>', '</s1>', '</s2>', '</s3>', '</s4>', '</s5>', '</s6>', '</s7>',
                                          '</s8>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.single:
            concated_sen = concate_sen(self.input, self.speaker, self.tokenizer)
            self.sent_ids, self.masks, self.token_types = tokenize(concated_sen, self.tokenizer, model_type=model_type)

        else:
            self.sent_ids, self.masks, self.token_types = tokenize_separate(self.input, self.tokenizer, self.speaker, model_type=model_type)

            # concated_sen_pre = concate_sen_pre(self.input, self.speaker, self.tokenizer)
            # self.sent_ids, self.masks, self.token_types = tokenize_pre(concated_sen_pre, self.tokenizer, self.speaker)

        self.len = len(self.keys)

        print('\n dataloader initialized!')

    # def __getitem__(self, index):
    #     conv = self.index_map[index]
    #
    #     return torch.LongTensor(self.sent_ids[conv]), \
    #            torch.LongTensor(self.masks[conv]), \
    #            torch.LongTensor(self.token_types[conv]), \
    #            self.speaker[conv], \
    #            self.speaker_matrix[conv], \
    #            self.turn_matrix[conv], \
    #            torch.FloatTensor([1] * self.label[conv].size(-1)), \
    #            self.label[conv], \
    #            self.type_label[conv], \
    #            self.label_td[conv], \
    #            self.type_label_td[conv], \
    #            self.turn[conv], \
    #            conv
    def __getitem__(self, index):
        conv = self.index_map[index]

        return torch.LongTensor(self.sent_ids[conv]), \
               torch.LongTensor(self.masks[conv]), \
               torch.LongTensor(self.token_types[conv]), \
               self.speaker[conv], \
               torch.FloatTensor([1] * self.label[conv].size(-1)), \
               self.label[conv], \
               self.type_label[conv], \
               self.label_td[conv], \
               self.type_label_td[conv], \
               torch.LongTensor(self.src_sp[conv]), \
               torch.LongTensor(self.dst_sp[conv]), \
               torch.LongTensor(self.src_turn[conv]), \
               torch.LongTensor(self.dst_turn[conv]), \
               self.turn[conv], \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)

        return [pad_sequence([dat[i][0]]) if i < 9 else dat[i][0] if i < 14 else dat[i].tolist() for i in dat]
        # return [pad_sequence([dat[i][0]]) if i < 11 else dat[i][0] if i < 12 else dat[i].tolist() for i in dat]
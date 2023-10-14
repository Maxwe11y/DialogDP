from dataloader_ta import MELDDataset

import pickle, json
from llm.llm import get_prompt, gen_response
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from sklearn.metrics import classification_report

def metrics(path=r'./llm/molweni_6/'):
    # emos = {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6}
    # emos = {'joyful': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sad': 4, 'powerful': 5, 'scared': 6}
    emos =  {'neutral': 0, 'positive': 1, 'negative': 2}
    with open(os.path.join(path, r'emotions.json'), 'r') as f:
        emotions = json.load(f)
    with open(os.path.join(path, r'responses.json'), 'r') as f:
        responses = json.load(f)
    res = []
    count = 0
    for dial in responses:
        for response in dial:
            pred_label = 3
            for emo, label in emos.items():
                if emo.lower() in response.lower():
                    pred_label = label
            res.append(pred_label)

    true_labels = [i for dial in emotions for i in dial ]
    # true_labels_ = []
    # res_ = []
    # for i, (pred_label, true_label) in enumerate(zip(res, true_labels)):
    #     if pred_label == -1:
    #         count+=1
    #     else:
    #         res_.append(pred_label)
    #         true_labels_.append(true_label)
            # if true_label >= 0 and true_label < 6:
            #     res[i] = true_label + 1
            # else:
            #     res[i] = true_label - 2
    result = classification_report(true_labels, res, labels=[0,1,2,3], digits=4)
    print(result)


def emory_to_dp(downstream_data='Emory', ddp_data='molweni'):
    data = pickle.load(open(r'./data/{}_revised/{}_{}_6.pkl'.format(downstream_data, downstream_data, ddp_data), 'rb'))
    speakers_, emotion_labels_, sentences_, trainId, testId, validId, structure_ = data
    dp_train = downstream2dp(trainId, sentences_, speakers_, structure_, emotion_labels_)
    dp_valid = downstream2dp(validId, sentences_, speakers_, structure_, emotion_labels_)
    dp_test = downstream2dp(testId, sentences_, speakers_, structure_, emotion_labels_)

    return dp_train, dp_valid, dp_test

def meld_to_dp(downstream_data='MELD', ddp_data='molweni'):
    # data = pickle.load(open(r'./data/{}_revised/{}_{}_16.pkl'.format(downstream_data, downstream_data, ddp_data), 'rb'))
    data = pickle.load(open(r'./data/{}_revised/{}_{}_6_pred_all_freeze.pkl'.format(downstream_data, downstream_data, ddp_data), 'rb'))
    videoIDs_, videoSpeakers_, videoLabels_, videoText_, videoAudio_, videoSentence_, trainVid, testVid, videoLabels_3, structure_, action_ = data
    # [trainVid_, validVid] = train_test_split(list(trainVid), test_size=0.1)
    # dp_train = downstream2dp(trainVid_, videoSentence_, videoSpeakers_, structure_)
    # dp_valid = downstream2dp(validVid, videoSentence_, videoSpeakers_, structure_)
    dp_test = downstream2dp(testVid, videoSentence_, videoSpeakers_, structure_, videoLabels_)

    return dp_test


def downstream2dp(dataid, sentences, speakers, structures, emotion_labels_):
    dataset = {}
    for k in dataid:
        texts = sentences[k]
        speakers_ = speakers[k]
        emos = emotion_labels_[k]
        dial = {}
        lines = []
        for idx, (text, speaker, emo) in enumerate(zip(texts, speakers_, emos)):
            text_toks = text.split(' ')
            line = {'speaker': str(speaker.index(1)), 'text': text, 'text_raw': text,
                    'tokens': text_toks, 'emo':emo, 'turn': idx+1}
            lines.append(line)
        dial['id'] = k
        dial['edus'] = lines
        dial['relations'] = [{'type':line['type'], 'x':line['x'], 'y':line['y']} for line in structures[k]]
        dial['labels'] = emos
        dataset[k] = dial

    return dataset


class LLMData(Dataset):
    def __init__(self, data):
        self.utts = {}
        self.his = {}
        self.structure_x = {}
        self.structure_y = {}
        self.keys = []
        self.emotions = {}
        idx = 0
        for k, dial in data.items():
            relations = dial['relations']
            rels_x = {}
            rels_y = {}
            for rel in relations:
                if rel['x'] not in rels_x:
                    rels_x[rel['x']] = {'y': [rel['y']], 'dep_type': [rel['type']]}
                else:
                    if rel['y'] not in rels_x[rel['x']]['y']:
                        rels_x[rel['x']]['y'].append(rel['y'])
                        rels_x[rel['x']]['dep_type'].append(rel['type'])
                if rel['y'] not in rels_y:
                    rels_y[rel['y']] = {'x': [rel['x']], 'head_type': [rel['type']]}
                else:
                    if rel['x'] not in rels_y[rel['y']]['x']:
                        rels_y[rel['y']]['x'].append(rel['x'])
                        rels_y[rel['y']]['head_type'].append(rel['type'])
            self.structure_x[idx] = rels_x
            self.structure_y[idx] = rels_y
            self.utts[idx] = [line['text'] for line in dial['edus']]
            self.emotions[idx] = dial['labels']
            self.keys.append(idx)
            idx += 1


        self.len = len(self.keys)
    def __getitem__(self, index):
        idx = self.keys[index]
        his = "\n".join(self.utts[idx])
        u_i = self.utts[idx]
        rel_head = {}
        head_utts = {}
        rel_dep ={}
        dep_utts = {}
        for i in range(len(u_i)):
            rel_head[i] = []
            head_utts[i] = []
            rel_dep[i] = []
            dep_utts[i] = []
            if i in self.structure_y[idx]:
                if 'head_type' in self.structure_y[idx][i]:
                    rel_head[i] = [tp for tp in self.structure_y[idx][i]['head_type']]
                if 'x' in self.structure_y[idx][i]:
                    head_utts[i] = [u_i[head] for head in self.structure_y[idx][i]['x']]
            if i in self.structure_x[idx]:
                if 'dep_type' in self.structure_x[idx][i]:
                    rel_dep[i] = [tp for tp in self.structure_x[idx][i]['dep_type']]
                if 'y' in self.structure_x[idx][i]:
                    dep_utts[i] = [u_i[head] for head in self.structure_x[idx][i]['y']]


        # rel_head = [[tp  for tp in self.structure[idx][i]['head_type']] for i in range(len(u_i)) if i in self.structure[idx] if ['head_type'] in self.structure[idx][i]]
        # head_utts = [[u_i[head] for head in self.structure[idx][i]['x']] for i in range(len(u_i)) if
        #             i in self.structure[idx] if ['x'] in self.structure[idx][i]]
        # rel_dep = [[tp for tp in self.structure[idx][i]['dep_type']] for i in range(len(u_i))  if
        #             i in self.structure[idx] if ['dep_type'] in self.structure[idx][i]]
        # dep_utts = [[u_i[head] for head in self.structure[idx][i]['y']] for i in range(len(u_i))  if
        #              i in self.structure[idx] if ['y'] in self.structure[idx][i]]
        emos = self.emotions[idx]

        return his, u_i, rel_head, head_utts, rel_dep, dep_utts, emos

    def __len__(self):
        return self.len

def generate_chatgpt_inp(task='Sentiment Analysis'):
    """his = "Yes, and it is my dying wish to have that ring.\nSee, if I'm not buried with that ring then my spirit is going to wander the nether world for all eternity",
        u_i="Okay, that's enough honey!",
        rel_head=['Acknowledgement'],
        head_utts=["See, if I'm not buried with that ring then my spirit is going to wander the nether world for all eternity"],
        rel_dep=['Comment'],
        dep_utts=["I don't know.  Let me see the ring."]"""
    data = meld_to_dp(downstream_data='MELD', ddp_data='molweni')
    # _, _, data = emory_to_dp(downstream_data='Emory', ddp_data='molweni')
    data_ = LLMData(data)
    emotions_ = []
    responses = []
    for dat in tqdm(data_):
        his, u_i, rel_head, head_utts, rel_dep, dep_utts, emos = dat
        emotions_.append(emos)
        response = []
        for idx, u in enumerate(u_i):

            prompt = get_prompt(
                his=his,
                u_i=u,
                rel_head=rel_head[idx],
                head_utts=head_utts[idx],
                rel_dep=rel_dep[idx],
                dep_utts=dep_utts[idx])

            res = gen_response(prompt, task=task, engine="gpt-3.5-turbo-0301")
            response.append(res)
        responses.append(response)
    with open('./llm/responses.json', 'w') as f:
        json.dump(responses, f)
    with open('./llm/emotions.json', 'w') as f:
        json.dump(emotions_, f)

# generate_chatgpt_inp(task='SA')

metrics(path='./llm/molweni_chagpt_3/')


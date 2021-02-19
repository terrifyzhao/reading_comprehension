import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import joblib
import json
from sklearn.utils import shuffle
import jieba


def json2csv(name, is_shuffle=False):
    data = json.load(open(f'data/{name}.json'))

    data_list = []

    for d in data:
        content = d['Content']
        # c_id = d['ID']
        questions = d['Questions']
        for question in questions:
            q = question['Question']
            choices = question['Choices']
            answer = question['Answer']
            q_id = question['Q_id']
            for choice in choices:
                if answer in choice:
                    data_list.append((q_id, content, q, choice, 1))
                else:
                    data_list.append((q_id, content, q, choice, 0))

    df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'answer'])
    if is_shuffle:
        df = shuffle(df)
    length = int(len(df) * 0.9)
    train = df[0:length]
    valid = df[length:]
    train.to_csv('data/train.csv', index=False, encoding='utf_8_sig')
    valid.to_csv('data/valid.csv', index=False, encoding='utf_8_sig')


class InputFeature:
    def __init__(self, input_ids, token_type_ids, attention_mask):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, encodings):
        self.labels = labels
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def process(name, tokenizer, batch_size, max_length=128, cut=False):
    if not os.path.exists(f'data/{name}.csv'):
        json2csv(name)

    if os.path.exists(f'data/{name}_dataset'):
        dataset = joblib.load(f'data/{name}_dataset')
    else:
        df = pd.read_csv(f'data/{name}.csv')[0:20]

        batch_input_ids = None
        batch_token_type_ids = None
        batch_attention_masks = None
        labels = []
        for tmp_df in tqdm(df.groupby('q_id')):
            label = None
            qas = []
            contents = []
            for i, row in enumerate(tmp_df[1].values[0:4]):
                qa = row[2] + ',' + row[3][2:]
                content = row[1]
                if row[4] == 1 and label is None:
                    label = i
                if cut:
                    qa = list(jieba.cut(qa))
                    content = list(jieba.cut(content))
                qas.append(qa)
                contents.append(content)
            encoding = tokenizer(qas,
                                 contents,
                                 return_tensors='pt',
                                 is_split_into_words=cut,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=max_length)

            if batch_input_ids is not None:
                batch_input_ids = torch.cat([batch_input_ids, encoding['input_ids']], dim=0)
            else:
                batch_input_ids = encoding['input_ids']
            if batch_token_type_ids is not None:
                batch_token_type_ids = torch.cat([batch_token_type_ids, encoding['token_type_ids']], dim=0)
            else:
                batch_token_type_ids = encoding['token_type_ids']
            if batch_attention_masks is not None:
                batch_attention_masks = torch.cat([batch_attention_masks, encoding['attention_mask']], dim=0)
            else:
                batch_attention_masks = encoding['attention_mask']

            while len(encoding['input_ids']) < 4:
                length = encoding['input_ids'].shape[1]

                batch_input_id = torch.cat([encoding['input_ids'], torch.zeros((1, length))], dim=0)
                if batch_input_ids is not None:
                    batch_input_ids = torch.cat([batch_input_ids, batch_input_id], dim=0)
                else:
                    batch_input_ids = batch_input_id

                batch_token_type_id = torch.cat([encoding['token_type_ids'], torch.zeros((1, length))], dim=0)
                if batch_token_type_ids is not None:
                    batch_token_type_ids = torch.cat([batch_token_type_ids, batch_token_type_id], dim=0)
                else:
                    batch_token_type_ids = batch_token_type_id

                batch_attention_mask = torch.cat([encoding['attention_mask'], torch.zeros((1, length))], dim=0)
                if batch_attention_masks is not None:
                    batch_attention_masks = torch.cat([batch_attention_masks, batch_attention_mask], dim=0)
                else:
                    batch_attention_masks = batch_attention_mask

                if label is None:
                    label = 3

            labels.append(label)
        dic = {'input_ids': batch_input_ids,
               'token_type_ids': batch_token_type_ids,
               'attention_mask': batch_attention_masks}

        dataset = Dataset(labels, dic)
        joblib.dump(dataset, f'data/{name}_dataset')

    loader = DataLoader(dataset, batch_size=batch_size)

    return loader


if __name__ == '__main__':
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('./bert')
    data = process('valid', tokenizer, batch_size=8)
    for d in data:
        print(d)
        break
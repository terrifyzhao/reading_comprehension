import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import joblib
import json
from sklearn.utils import shuffle


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


def process(name, tokenizer, batch_size, max_length=128, predict=False):
    if not os.path.exists(f'data/{name}.csv'):
        json2csv(name)

    if os.path.exists(f'data/{name}_dataset'):
        dataset = joblib.load(f'data/{name}_dataset')
    else:
        df = pd.read_csv(f'data/{name}.csv')

        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        labels = []

        for tmp_df in tqdm(df.groupby('q_id')):
            batch_input_id = []
            batch_token_type_id = []
            batch_attention_mask = []
            label = None
            for i, row in enumerate(tmp_df[1].values[0:4]):
                qa = row[2] + ',' + row[3][2:]
                content = row[1]
                if row[4] == 1 and label is None:
                    label = i
                encoding = tokenizer(qa,
                                     content,
                                     return_tensors='pt',
                                     truncation=True,
                                     padding='max_length',
                                     max_length=max_length)
                batch_input_id.append(encoding['input_ids'])
                batch_token_type_id.append(encoding['token_type_ids'])
                batch_attention_mask.append(encoding['attention_mask'])
            while len(batch_input_id) < 4:
                batch_input_id.append(torch.zeros((1, max_length)))
                batch_token_type_id.append(torch.zeros((1, max_length)))
                batch_attention_mask.append(torch.zeros((1, max_length)))
                if label is None:
                    label = 3

            labels.append(label)
            batch_input_ids.append(torch.cat(batch_input_id, dim=0))
            batch_token_type_ids.append(torch.cat(batch_token_type_id, dim=0))
            batch_attention_masks.append(torch.cat(batch_attention_mask, dim=0))

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

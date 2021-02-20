import json
from transformers import BertTokenizerFast, BertForMultipleChoice
import torch
import pandas as pd
from tqdm import tqdm
from snippets import process

batch_size = 100
model_path = '/home/joska/ptm/roberta'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = json.load(open('data/validation.json'))

# pre
data_list = []
for d in data:
    content = d['Content']
    questions = d['Questions']
    for question in questions:
        q = question['Question']
        choices = question['Choices']
        q_id = question['Q_id']
        for choice in choices:
            data_list.append((q_id, content[0:512], q, choice, 1))

df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'labels'])
df.to_csv('data/test_pre.csv', index=False, encoding='utf_8_sig')

# mid
data_list = []
for d in data:
    content = d['Content']
    questions = d['Questions']
    for question in questions:
        q = question['Question']
        choices = question['Choices']
        q_id = question['Q_id']
        for choice in choices:
            tmp_content = content[0:256] + content[-256:]
            if tmp_content:
                data_list.append((q_id, tmp_content, q, choice, 1))
            else:
                data_list.append((q_id, content, q, choice, 1))

df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'labels'])
df.to_csv('data/test_mid.csv', index=False, encoding='utf_8_sig')

# post
data_list = []
for d in data:
    content = d['Content']
    questions = d['Questions']
    for question in questions:
        q = question['Question']
        choices = question['Choices']
        q_id = question['Q_id']
        for choice in choices:
            data_list.append((q_id, content[-512:], q, choice, 1))

df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'labels'])
df.to_csv('data/test_post.csv', index=False, encoding='utf_8_sig')

tokenizer = BertTokenizerFast.from_pretrained(model_path)
# model = BertForMultipleChoice.from_pretrained(model_path)
model = torch.load('roberta.bin').to(device)

test_loader_pre = process('test_pre', tokenizer, batch_size, max_length=512, cut=False)
test_loader_mid = process('test_mid', tokenizer, batch_size, max_length=512, cut=False)
test_loader_post = process('test_post', tokenizer, batch_size, max_length=512, cut=False)

output = []
for batch_pre, batch_mid, batch_post in tqdm(zip(test_loader_pre, test_loader_mid, test_loader_post)):
    # for batch_pre in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch_pre['input_ids'].long().to(device)
        attention_mask = batch_pre['attention_mask'].to(device)
        outputs_pre = model(input_ids, attention_mask=attention_mask)

        input_ids = batch_mid['input_ids'].long().to(device)
        attention_mask = batch_mid['attention_mask'].to(device)
        outputs_mid = model(input_ids, attention_mask=attention_mask)

        input_ids = batch_post['input_ids'].long().to(device)
        attention_mask = batch_post['attention_mask'].to(device)
        outputs_post = model(input_ids, attention_mask=attention_mask)

        logits = (outputs_pre.logits + outputs_mid.logits + outputs_post.logits) / 3
        # logits = outputs_pre.logits
        output.extend(logits.argmax(dim=1).cpu().numpy())

result = pd.DataFrame()
result['id'] = df['q_id'].unique()
result['label'] = output


def label_mapping(x):
    if x == 0:
        return 'A'
    elif x == 1:
        return 'B'
    elif x == 2:
        return 'C'
    elif x == 3:
        return 'D'


result['label'] = result['label'].apply(lambda x: label_mapping(x))
result.to_csv('result.csv', index=False)

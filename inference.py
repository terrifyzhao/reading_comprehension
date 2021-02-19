import json
from transformers import BertTokenizerFast
import torch
import pandas as pd
from tqdm import tqdm
from process import process

batch_size = 100
model_path = './bert'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = json.load(open('data/validation.json'))

data_list = []
for d in data:
    content = d['Content']
    questions = d['Questions']
    for question in questions:
        q = question['Question']
        choices = question['Choices']
        q_id = question['Q_id']
        for choice in choices:
            data_list.append((q_id, content, q, choice, 1))

df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'labels'])
df.to_csv('data/test.csv', index=False, encoding='utf_8_sig')

tokenizer = BertTokenizerFast.from_pretrained(model_path)
test_loader = process('test', tokenizer, batch_size, max_length=512, cut=True)

model = torch.load('best_model.bin').to(device)

output = []
for batch in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].long().to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        output.extend(outputs.logits.argmax(dim=1).cpu().numpy())

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



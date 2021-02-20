from transformers import BertTokenizerFast, AdamW
from transformers import BertForMultipleChoice, get_linear_schedule_with_warmup, XLNetForMultipleChoice
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from process import process
import os

# content 序列长度，均值: 1031.635108801559 80%分位数: 1288.0 90%分位数: 1482.0
# question 序列长度，均值: 24.665890801486775 80%分位数: 28.0 90%分位数: 32.0
# choice 序列长度，均值: 43.36496337194616 80%分位数: 58.0 90%分位数: 64.0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 4
lr = 2e-5
max_length = 512
accumulation_steps = 4
model_path = '/home/joska/ptm/bert_large'
min_valid_loss = 1.2289

tokenizer = BertTokenizerFast.from_pretrained(model_path)

print('train loader')
train_loader = process('train', tokenizer, batch_size, max_length=max_length, cut=False)
print('valid loader')
valid_loader = process('valid', tokenizer, batch_size, max_length=max_length, cut=False)

if os.path.exists('best_model.bin'):
    print('load model')
    model = torch.load('best_model.bin')
else:
    model = BertForMultipleChoice.from_pretrained(model_path)
model.to(device)

optim = AdamW(model.parameters(), lr=lr)
# warm_up = get_linear_schedule_with_warmup(optim,
#                                           num_warmup_steps=500,
#                                           num_training_steps=len(train_loader) * 3)


def train_func():
    global min_valid_loss
    train_loss = 0
    train_acc = 0
    pbar = tqdm(train_loader)
    i = 0
    for batch in pbar:
        i += 1
        # optim.zero_grad()
        input_ids = batch['input_ids'].long().to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].long().to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        output = outputs.logits

        train_loss += loss.item()
        loss = loss / accumulation_steps
        loss.backward()

        if (i % accumulation_steps) == 0:
            # optimizer the net
            optim.step()  # update parameters of net
            optim.zero_grad()  # reset gradient

        label = labels.cpu().numpy()
        output = output.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(label, output)
        train_acc += acc

        pbar.update()
        pbar.set_description(f'loss:{loss.item() * accumulation_steps:.4f}, acc:{acc:.4f}')

        if i % 400 == 0:
            valid_loss, valid_acc = test_func()
            print(f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(model, 'best_model.bin')
                print('save model done')

    return train_loss / len(train_loader), train_acc / len(train_loader)


def test_func():
    valid_loss = 0
    valid_acc = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].long().to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].long().to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            output = outputs.logits
            valid_loss += loss.item()
            label = labels.cpu().numpy()
            output = output.argmax(dim=1).cpu().numpy()
            valid_acc += accuracy_score(label, output)
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)


for epoch in range(5):
    print('************start train************')
    train_loss, train_acc = train_func()
    print(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
    print('************start valid************')
    valid_loss, valid_acc = test_func()
    print(f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')

    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, 'best_model.bin')
        print('save model done')

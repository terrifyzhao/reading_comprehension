from transformers import BertTokenizerFast, AdamW
from transformers import BertForMultipleChoice, get_cosine_schedule_with_warmup
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from snippets import process, fix_seed, FGM
import os

# content 序列长度，均值: 1031.635108801559 80%分位数: 1288.0 90%分位数: 1482.0
# question 序列长度，均值: 24.665890801486775 80%分位数: 28.0 90%分位数: 32.0
# choice 序列长度，均值: 43.36496337194616 80%分位数: 58.0 90%分位数: 64.0


fix_seed(2021)
batch_size = 12
lr = 2e-5
max_length = 512
epochs = 4
accumulation_steps = 4
base_path = '/home/joska/ptm/'
model_name = 'roberta'
model_path = base_path + model_name
# model_path = './roberta'
min_valid_loss = float('inf')
use_fgm = True
use_gpu = torch.cuda.is_available()
use_multi_gpu = False and use_gpu
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

tokenizer = BertTokenizerFast.from_pretrained(model_path)

print('train loader')
train_loader = process('train', tokenizer, batch_size, max_length=max_length)
print('valid loader')
valid_loader = process('valid', tokenizer, batch_size, max_length=max_length)

if os.path.exists(f'{model_name}.bin'):
    print('load model')
    model = torch.load(f'{model_name}.bin')
else:
    model = BertForMultipleChoice.from_pretrained(model_path)

if use_multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
if use_gpu:
    model.cuda()

optim = AdamW(model.parameters(), lr=lr)

num_training_steps = len(train_loader) * epochs // accumulation_steps
# num_warmup_steps = num_training_steps * 0.1 // accumulation_steps
num_warmup_steps = 0
warm_up = get_cosine_schedule_with_warmup(optim,
                                          num_warmup_steps=num_warmup_steps,
                                          num_training_steps=num_training_steps)
fgm = FGM(model)


def train_func():
    global min_valid_loss
    train_loss = 0
    train_acc = 0
    pbar = tqdm(train_loader)
    i = 0
    for batch in pbar:
        i += 1

        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask']
        labels = batch['labels'].long()

        if use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        output = outputs.logits

        mean_loss = torch.mean(loss)
        train_loss += mean_loss.item()

        loss = loss / accumulation_steps
        loss.backward()

        if use_fgm:
            fgm.attack(emb_name='embeddings.word_embeddings.weight')
            loss_adv = model(input_ids, attention_mask=attention_mask, labels=labels).loss
            loss_adv.backward()
            fgm.restore(emb_name='embeddings.word_embeddings.weight')

        if (i % accumulation_steps) == 0:
            optim.step()
            optim.zero_grad()
            warm_up.step()

        label = labels.cpu().numpy()
        output = output.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(label, output)
        train_acc += acc

        pbar.update()
        pbar.set_description(f'loss:{loss.item() * accumulation_steps:.4f}, acc:{acc:.4f}')

        if i % (num_training_steps // 3 + 10) == 0:
            valid_loss, valid_acc = test_func()
            print(f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(model, f'{model_name}.bin')
                print('save model done')

    return train_loss / len(train_loader), train_acc / len(train_loader)


def test_func():
    valid_loss = 0
    valid_acc = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].long()
            attention_mask = batch['attention_mask']
            labels = batch['labels'].long()
            if use_gpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            output = outputs.logits
            valid_loss += loss.item()
            label = labels.cpu().numpy()
            output = output.argmax(dim=1).cpu().numpy()
            valid_acc += accuracy_score(label, output)
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)


for epoch in range(epochs):
    print('************start train************')
    train_loss, train_acc = train_func()
    print(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
    print('************start valid************')
    valid_loss, valid_acc = test_func()
    print(f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')

    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, f'{model_name}.bin')
        print('save model done')

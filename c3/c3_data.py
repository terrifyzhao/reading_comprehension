import json
import pandas as pd


def json2csv(name):
    data = json.load(open(f'c3-m-{name}.json'))

    data_list = []

    for i, d in enumerate(data):
        content = d[0][0]
        questions = d[1]
        for j, question in enumerate(questions):
            q = question['question']
            choices = question['choice']
            answer = question['answer']
            q_id = str(i) + '_' + str(j)
            for choice in choices:
                if answer in choice:
                    data_list.append((q_id, content, q, choice, 1))
                else:
                    data_list.append((q_id, content, q, choice, 0))

    df = pd.DataFrame.from_records(data_list, columns=['q_id', 'content', 'question', 'choice', 'answer'])
    df.to_csv(f'c3_{name}.csv', index=False, encoding='utf_8_sig')
    return df


if __name__ == '__main__':
    train_df = json2csv('train')
    train_dev = json2csv('dev')
    train_test = json2csv('test')
    train_df = train_df.append(train_dev)
    train_df.to_csv(f'c3_train.csv', index=False, encoding='utf_8_sig')

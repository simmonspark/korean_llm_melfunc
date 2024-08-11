import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

abs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
spesific_path = 'korean_lang'
data_dir = os.path.join(abs_path, spesific_path)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eos_ids = tokenizer.eos_token_id
print("EOS Token ID:", eos_ids)
tokenizer.pad_token = tokenizer.eos_token

def get_data(path: str):
    '''
        :param: path
        :return: raw text data
    '''
    full = []
    for dir, _, path_list in os.walk(path):
        for name in path_list:
            full_path = os.path.join(dir, name)
            full.append(full_path)
    return full


def preprocessing(path_list):
    assert type(path_list) == list, "Input should be a list of file paths."

    input_data = []
    labels = []
    print('---processing raw json as input...\n')
    for path in tqdm(path_list):
        with open(path, 'r', encoding='utf-8') as file:
            raw_json = json.load(file)

            for dialogue in raw_json['data']:
                summary = dialogue['body']['summary']
                tmp = []
                for utterance in dialogue['body']['dialogue']:
                    tmp.append([utterance['utterance']])
                labels.append(summary)
                input_data.append(tmp)

    context = []

    for sequence in input_data:
        tmp = ''
        for word in sequence:
            tmp += word[0]
        context.append(tmp)

    train_x, test_x, train_y, test_y = train_test_split(context,labels,train_size=0.8,shuffle=True)
    print(f'vocab_size is {tokenizer.vocab_size}')
    tokenizer_test('내 이름은 시언')
    return train_x, test_x, train_y, test_y
def toknizing(context,mode = 'train'):

    return tokenizer(context,truncation=True, max_length=1024, return_tensors='pt',padding='max_length')

def tokenizer_test(sequence):
    before = sequence
    embedded = tokenizer(before, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
    input_ids = embedded['input_ids']
    decode = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    tmp = embedded['input_ids']
    print(f'before tokenizing : {tmp}')
    print(f'after decode : {decode}')

    assert before == decode, f"Original: {before}, Decoded: {decode}"
    print('\nencode <-> decode test passed! || ... waitting after process ...\n')

def decode(Y):
    decode = tokenizer.decode(Y[0], skip_special_tokens=True) if type(Y) == list else tokenizer.decode(Y, skip_special_tokens=True)
    return decode

if __name__ == "__main__":
    train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))

    print()
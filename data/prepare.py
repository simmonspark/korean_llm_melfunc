import os
import json
from datasets import Dataset
import numpy as np
import tiktoken
from tqdm import tqdm

#data inform https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71630

toknizer = tiktoken.get_encoding('gpt2')

bs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
specific_path = 'korean_lang/Training/data_train/data'
data_dir = os.path.join(bs_path, specific_path)

data_files = ['개인및관계.json', '미용과건강.json', '시사교육.json', '식음료.json', '여가생활.json', '주거와생활.json', '일과직업.json',
              '상거래(쇼핑).json']


def toknizing(data):
    ids = toknizer.encode_ordinary(data['dialogue'])
    ids.extend(toknizer.encode_ordinary('[start]'))
    ids.extend(toknizer.encode_ordinary(data['summary']))
    ids.append(toknizer.eot_token)
    return {'ids': ids, 'len': len(ids)}


def extract_dialogues_and_summaries(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dialogues = []
    for dialogue in raw_data['data']:
        dialogue_text = ' '.join([utterance['utterance'] for utterance in dialogue['body']['dialogue']])
        summary_text = dialogue['body']['summary']
        dialogues.append({'dialogue': dialogue_text, 'summary': summary_text})
    return dialogues


all_dialogues = []
for file in data_files:
    file_path = os.path.join(data_dir, file)
    all_dialogues.extend(extract_dialogues_and_summaries(file_path))

dataset = Dataset.from_list(all_dialogues)

if __name__ == '__main__':
    dataset = dataset.train_test_split(test_size=0.05, seed=1234, shuffle=True)

    encoded_train = dataset['train'].map(function=toknizing, remove_columns=['dialogue', 'summary'], desc='열심히 토큰화!',
                                         num_proc=12)
    encoded_val = dataset['test'].map(function=toknizing, remove_columns=['dialogue', 'summary'], desc='토크나이징 노오력중!',
                                      num_proc=12)

    print(f"Encoded training dataset: {encoded_train}")
    print(f"Encoded validation dataset: {encoded_val}")
    enced_list = [encoded_train, encoded_val]
    print(encoded_train['len'])
    for split, dset in enumerate(enced_list):
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        tmp = 'train' if split == 0 else 'eval'
        filename = os.path.join(os.path.dirname(__file__), f'{tmp}.bin')
        dtype = np.uint16
        # numpy 메모리맵을 만들어놈
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 512
        idx = 0
        # 데이터 샤딩에 들어감
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # 데이터 샤딩해서 dataset 라이브러리에서 지원하는 numpy format으로 만들어버림
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            # 배치에 있는 인덱스를 한줄로 정렬 (samples * len,) 형태임
            arr_batch = np.concatenate(batch['ids'])

            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


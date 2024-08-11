from transformers import AutoTokenizer
import os
import time
from contextlib import nullcontext
from transformers import BertModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import Korean_Dataset
from model import GPTConfig, GPT
from config import GPT_CONFIG
from utils import preprocessing, get_data, decode
from torch.utils.data import DataLoader
import os

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eos_ids = tokenizer.eos_token_id
print("EOS Token ID:", eos_ids)
tokenizer.pad_token = tokenizer.eos_token
# 모델과 토크나이저 로드

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cfg = GPT_CONFIG()
data_dir = 'data/openwebtext'  # 데이터 디렉토리 경로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.embedding_depth, block_size=cfg.block_size,
                  bias=cfg.bias, vocab_size=50304, dropout=cfg.dropout)
torch.manual_seed(1337 + cfg.seed_offset)
torch.backends.cuda.matmul.allow_tf32 = cfg.torch_16bit_allow
torch.backends.cudnn.allow_tf32 = cfg.torch_allow_tf32

print('======================================')
print('       언시 GPT TRAINING PROCESS      ')
print('======================================\n')
print("Initializing a new model from scratch")
print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)\n")
print('---- Config as follow ----\n')
print(cfg)


def get_batch(loader, device):
    for batch in loader:
        input_ids = batch['input_ids'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        yield input_ids, labels


def load_model(checkpoint_path, cfg, device):
    cfg = GPT_CONFIG()
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.embedding_depth, block_size=cfg.block_size,
                      bias=cfg.bias, vocab_size=50304, dropout=cfg.dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # 체크포인트 로드
    state_dict = torch.load(checkpoint_path)['model']

    # 키 이름 수정: '_orig_mod.' 접두사 제거
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    # 수정된 state_dict를 모델에 로드
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    return model


scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))
gptconf = GPTConfig(**model_args)
print(model_args)
model = GPT(gptconf)
model = model.to(cfg.device)
model = load_model('./out/ckpt.pt', cfg, 'cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to('cuda')
abs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
spesific_path = 'korean_lang'
data_dir = os.path.join(abs_path, spesific_path)
eos_token = '<|endoftext|>'
eos_ids = tokenizer.add_special_tokens({'eos_token': eos_token})

train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))
train_dataset = Korean_Dataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

X, Y = next(get_batch(train_loader, device))
'''
X = '금요일에 놀가야지. 도토리 숲도 가고. 사진도 찍고. 아 맞다 키링 꼭 사야겠어'
X = tokenizer.encode(X)
x = (torch.tensor(X, dtype=torch.long, device=device)[None, ...]).to('cuda')
'''

with torch.no_grad():
    for k in range(10):
        y = model.generate(X, max_new_tokens = 100, temperature=1.0, top_k=200)
        print(decode(y[0].tolist()))
        print('---------------')

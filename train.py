import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT
from config import GPT_CONFIG

cfg = GPT_CONFIG()
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
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

scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

'''model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.embedding_depth, block_size=cfg.block_size,
                  bias=cfg.bias, vocab_size=50257, dropout=cfg.dropout)
gptconf = GPTConfig(**model_args)
print(model_args)
model = GPT(gptconf)
model = model.to(cfg.device)
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import Korean_Dataset
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-small")
#model_custom_config = model.config
model = model.to('cuda')

# cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.device
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=[cfg.beta1, cfg.beta2])
train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))

train_ds = Korean_Dataset(train_x, train_y)
val_ds = Korean_Dataset(test_x, test_y)

train_loader = DataLoader(train_ds, batch_size=6, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=6, pin_memory=True, num_workers=8)
if cfg.compile:
    print("compiling the model... (시간이 좀 걸려요..)")
    unoptimized_model = model
    model = torch.compile(model)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype)


@torch.no_grad()
def cal_loss():
    val_iter = 0
    model.eval()
    losses = []
    for dic in tqdm(val_loader):
        x = dic['input_ids'].to('cuda')
        y = dic['labels'].to('cuda')
        attention_mask = dic['attention_mask'].to('cuda')
        with torch.cuda.amp.autocast(enabled=(cfg.dtype == 'float16')):
            outputs = model(input_ids=x, labels=y, attention_mask=attention_mask)
            loss = outputs.loss
        losses.append(loss)
        val_iter += 1
        if val_iter == 50:
            model.train()
            print(f'val loss : {sum(losses) / len(losses)}')
            return sum(losses) / len(losses)
    model.train()
    print(f'val loss : {sum(losses) / len(losses)}')
    return sum(losses) / len(losses)


i = 0
while True:
    i += 1
    g_loss = []
    for dic in tqdm(train_loader):
        x = dic['input_ids'].to('cuda')
        y = dic['labels'].to('cuda')
        attention_mask = dic['attention_mask'].to('cuda')
        with torch.cuda.amp.autocast(enabled=(cfg.dtype == 'float16')):
            outputs = model(input_ids=x, labels=y, attention_mask=attention_mask)
            loss = outputs.loss
        scaler.scale(loss).backward()
        if cfg.gradient_clipping > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        g_loss.append(loss.item())
    print(f"iter {i}: loss {sum(g_loss) / len(g_loss):.4f}")

    val_loss = cal_loss()
    print(f"Validation loss: {val_loss:.4f}")
    if cfg.save_model:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(cfg.out_dir, 'checkpoint.pt'))

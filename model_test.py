import os
import time
from contextlib import nullcontext
from transformers import BertModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import Korean_Dataset
from model import GPTConfig, GPT
from config import GPT_CONFIG
from utils import preprocessing, get_data
from torch.utils.data import DataLoader
import os



cfg = GPT_CONFIG()
data_dir = 'data/openwebtext'  # 데이터 디렉토리 경로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.embedding_depth, block_size=cfg.block_size,
                  bias=cfg.bias, vocab_size=50304, dropout=cfg.dropout)
gptconf = GPTConfig(**model_args)
print(model_args)
model = GPT(gptconf)
model = model.to(cfg.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
optimizer = model.configure_optimizers(cfg.weight_decay, 1e-6, (cfg.beta1, cfg.beta2), cfg.device)

if cfg.compile:
    print("compiling the model... (시간이 좀 걸려요..)")
    unoptimized_model = model
    model = torch.compile(model)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype)


def get_batch(loader, device):
    for batch in loader:
        input_ids = batch['input_ids'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        mask = batch['attention_mask'].squeeze(1).to(device)
        yield input_ids, labels, mask


@torch.no_grad()
def estimate_loss(loader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_interval)
        for k in range(cfg.eval_interval):
            X, Y, mask = next(get_batch(loader, device))
            with ctx:
                logits, loss = model(X, targets=Y, att_mask = mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


abs_path = '/media/sien/DATA/DATA/dataset/korean_summary'
spesific_path = 'korean_lang'
data_dir = os.path.join(abs_path, spesific_path)
train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))

train_dataset = Korean_Dataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

X, Y, mask = next(get_batch(train_loader, device))
t0 = time.time()
local_iter_num = 0
raw_model = model
running_mfu = -1.0

iter_num = 0
best_val_loss = 1e9
while True:

    lr = cfg.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % cfg.eval_interval == 0:
        losses = estimate_loss(train_loader)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or cfg.save_model:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {cfg.out_dir}")
                torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))

    for micro_step in range(cfg.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, targets=Y, att_mask = mask)
            loss = loss / cfg.gradient_accumulation_steps  # scale the loss to account for gradient accumulation

        X, Y, mask = next(get_batch(train_loader, device))

        scaler.scale(loss).backward()

    if cfg.gradient_clipping != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.log_interval == 0:
        lossf = loss.item() * cfg.gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.iter:
        break

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import Korean_Dataset
from utils import *
from torch.utils.data import DataLoader
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-small")
checkpoint = torch.load('out/checkpoint.pt')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to('cuda')
model.eval()

tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-small")

train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))
test_ds = Korean_Dataset(test_x, test_y)
train_ds = Korean_Dataset(train_x, train_y)
test_loader = DataLoader(train_ds, batch_size=10, pin_memory=True, num_workers=4)


def test_model():
    results = []
    dic = next(iter(test_loader))
    x = dic['input_ids'].to('cuda')
    attention_mask = dic['attention_mask'].to('cuda')

    with torch.cuda.amp.autocast(enabled=True):
        outputs = model.generate(input_ids=x, attention_mask=attention_mask)

    for _ in range(10):
        input_token = tokenizer.decode(x[_], skip_special_tokens=True)
        pred_text = tokenizer.decode(outputs[_], skip_special_tokens=True)
        target_text = tokenizer.decode(dic['labels'][_], skip_special_tokens=True)
        results.append((input_token, pred_text))

    return results


test_results = test_model()
for i, (input_token, pred_text) in enumerate(test_results):
    print(f"Example {i + 1}")
    print(f"입력은 다음과 같습니다.--->: {input_token}")
    print(f"문장 요약을 출력합니다.---> : {pred_text}\n")

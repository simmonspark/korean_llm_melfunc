import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# KoGPT-2 모델과 토크나이저 불러오기
tokenizer = GPT2Tokenizer.from_pretrained("taeminlee/kogpt2")
model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")

# 특수 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

# 입력 텍스트
text = "금요일에 아이파크몰에 가는데 너무 재밌겠다. 분위기 좋은 카페도 가고 아주고냥 지브리 기념품 파는 도토리숲도 가야지"

# 텍스트 토큰화
inputs = tokenizer(text, return_tensors="pt")

# Attention mask 설정
attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.long)

# 모델을 사용하여 요약 생성
with torch.no_grad():
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_new_tokens=50,   # 'max_length' 대신 'max_new_tokens' 사용
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # 패딩 토큰 ID 설정
    )

# 요약된 텍스트 디코드
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, errors="ignore")
print("Summary:", summary)

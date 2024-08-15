from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import BertProcessing


# 1. 토크나이저 초기화
tokenizer = Tokenizer(models.WordPiece(unk_token="[WARN]"))


# 2. Pre-tokenizer 설정 (한국어는 공백을 사용하여 어절 단위로 나눔)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 3. Trainer 설정 (한글에 특화된 어휘 크기 및 특수 토큰 설정)
trainer = trainers.WordPieceTrainer(
    vocab_size=53000, # 얼마나 많은 단어를 기억할거냐
    min_frequency=2, # 얼마나 자주 등장해야 vocab에 추가할건지
    special_tokens=["[WARN]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 4. 학습 데이터 준비
# dict 데이터에서 텍스트 데이터만 추출
data = {
    "key1": "최대한 긴 문장으로 예시를 작성해요~.",
    "key2": "자연어처리는 처음에 정말 재미가 없었지만 지금은 너무 재밌어요.",
    "key3": "Hugging Face는 자연어처리에 없어저는 안되는 필수 라이브러리에요."
}

# 데이터의 값들을 리스트로 변환
texts = list(data.values())

# 5. 토크나이저 학습
tokenizer.train_from_iterator(texts, trainer)

# 6. Post-processor 설정 (BERT와 같은 모델을 위한 CLS, SEP 처리)
tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)

#토크나이저 테스트
encoded = tokenizer.encode("Hugging Face는 정말 훌륭한 도구입니다.")
print(encoded.tokens)

#토크나이저 저장
tokenizer.save("korean_tokenizer.json")

# 학습된 어휘 사전 확인
print(tokenizer.get_vocab())


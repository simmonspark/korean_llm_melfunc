import torch
from transformers import BertTokenizer, BertModel

class BERT:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.model = BertModel.from_pretrained(config.model_name)
        self.model = self.model.to(config.device)

    def summarize(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config.max_length, truncation=True, padding='max_length').to(self.config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

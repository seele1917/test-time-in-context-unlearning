import re
import random
from functools import partial
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

class CustomEmbedding(torch.nn.Module):
    def __init__(self, embedding):
        super(CustomEmbedding, self).__init__()
        self.embedding = embedding
        # 勾配を制御するための勾配フックを設定
        self.embedding.weight.register_hook(self.grad_hook)

    def grad_hook(self, grad):
        # 更新するインデックスを定義
        indices_to_update = torch.tensor([32000, 32001], dtype=torch.long)
        # 勾配をゼロで初期化
        mask = torch.zeros_like(grad)
        # 指定したインデックスのみ勾配を保持
        mask[indices_to_update] = 1
        return grad * mask

    def forward(self, input):
        return self.embedding(input)

def get_model_and_tokenizer(model_name):
    token = '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'
    
    if model_name == 'llama2-7b':
        base_model_name = 'meta-llama/Llama-2-7b-chat-hf'
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            use_auth_token=token
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=token)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
    elif model_name == 'llama2-13b':
        base_model_name = 'meta-llama/Llama-2-13b-chat-hf'
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            use_auth_token=token
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=token)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        
    elif model_name == 'mistral-7b':
        base_model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            use_auth_token=token
        )
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_auth_token=token)
        tokenizer.pad_token = tokenizer.unk_token
    else:
        ValueError(f"{model_name} is not supported !!")
        

    
    # Define special tokens
    special_tokens = ["<<UNL>>", "<</UNL>>"]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


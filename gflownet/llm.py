import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_llm(model_name, low_cpu_mem_usage=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        low_cpu_mem_usage=low_cpu_mem_usage, 
        device_map="auto"
    )

    model.eval()

    return model

def get_last_hidden_layer(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    def mean_pool(hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = (hidden_states * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

    return mean_pool(last_hidden, inputs["attention_mask"])
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc

from pathlib import Path
import pickle
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`.*")

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.eval()

    return model

def get_last_hidden_layer(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

def embed_constraints(cfg):
    data_path = Path(__file__).parent.parent / "data"
    data_path = data_path / Path(cfg.input)  # string to pathlib.Path

    pickles = list(data_path.rglob("*.pickle"))

    consts = []
    files = []

    print("Collecting constraints...")
    for f in pickles:
        with open(f, 'rb') as p:
            x = pickle.load(p)
            assert 'constraint' in x.keys(), f"Missing constraint in {f}."
            if 'embedding' not in x.keys():
                files.append(f)
                consts.append(x['constraint'])

    if consts:
        tokenizer = get_tokenizer(cfg.llm)
        llm = get_llm(cfg.llm)

        print(f"Embedding {len(consts)} constraints in batches...")
        for i in tqdm(range(0, len(consts), cfg.llm_batch_size)):
            cbatch = consts[i:i+cfg.llm_batch_size]
            fbatch = files[i:i+cfg.llm_batch_size]

            ebatch = get_last_hidden_layer(cbatch, tokenizer, llm)

            for e, f in zip(ebatch, fbatch):
                with open(f, 'rb') as p:
                    x = pickle.load(p)

                x['embedding'] = e.cpu().float()

                with open(f, 'wb') as p:
                    pickle.dump(x, p, pickle.HIGHEST_PROTOCOL)

        del tokenizer
        del llm

    else:
        print("All constraints already embedded.")

    torch.cuda.empty_cache()
    gc.collect()
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc

from pathlib import Path
import pickle
from tqdm import tqdm

import networkx as nx

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`.*")

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_llama(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

def get_bert(model_name):
    model = AutoModel.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        output_hidden_states=True
    )

    return model

def get_llm(cfg):
    if "llama" in cfg.llm:
        model = get_llama(cfg.llm)

    elif "bert" in cfg.llm:
        model = get_bert(cfg.llm)
        model.to(cfg.device)

    else:
        raise ValueError(f"Model {cfg.llm} not supported. Please use BERT or LLaMA model from the Hugging Face Hub.")

    model.eval()

    return model

def get_last_llama_layer(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def last_non_padding_token(hidden_states, attention_mask):
        seq_lens = attention_mask.sum(dim=1)
        batch_size, _, hidden_dim = hidden_states.shape
        idx = (seq_lens - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, hidden_dim)

        return hidden_states.gather(1, idx).squeeze(1)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

    return last_non_padding_token(last_hidden, inputs["attention_mask"])

def get_last_bert_layer(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state[:,0,:]

    return last_hidden

def get_last_hidden_layer(prompts, tokenizer, model):
    if "llama" in model.name_or_path:
        return get_last_llama_layer(prompts, tokenizer, model)

    elif "bert" in model.name_or_path:
        return get_last_bert_layer(prompts, tokenizer, model)
    
    else:
        raise ValueError(f"Model {model.name_or_path} not supported. Please use BERT or LLaMA model from the Hugging Face Hub.")

def encode_nodes(cfg):
    if "tiny" in cfg.input:
        max_graph_size = 20
    elif "small" in cfg.input:
        max_graph_size = 300
    elif "large" in cfg.input:
        max_graph_size = 1200
    else:
        raise ValueError(f"Valid graph_type not specified in dataset name: {cfg.input}. Please use 'tiny', 'small', or 'large'.")
    node_identifiers = [f"this is node {i}" for i in range(max_graph_size)]

    tokenizer = get_tokenizer(cfg)
    llm = get_llm(cfg)

    node_encodings = []
    for i in tqdm(range(0, max_graph_size, cfg.llm_batch_size)):
        ebatch = get_last_hidden_layer(node_identifiers[i:i+cfg.llm_batch_size], tokenizer, llm)
        node_encodings.append(ebatch)
    
    return torch.cat(node_encodings, dim=0)

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
        print("Encoding node identifiers...")
        node_encodings = encode_nodes(cfg)

        tokenizer = get_tokenizer(cfg)
        llm = get_llm(cfg)

        print(f"Embedding {len(consts)} constraints in batches...")
        for i in tqdm(range(0, len(files), cfg.llm_batch_size)):
            Cbatch = consts[i:i+cfg.llm_batch_size]
            fbatch = files[i:i+cfg.llm_batch_size]

            cbatch = get_last_hidden_layer(Cbatch, tokenizer, llm)

            for c, f in zip(cbatch, fbatch):
                with open(f, 'rb') as p:
                    x = pickle.load(p)

                g = x['graph']
                nx.set_node_attributes(g, 
                    {i: node_encodings[i].cpu().float() for i in range(len(g))}, 
                    'encoding')

                x['embedding'] = c.cpu().float()

                with open(f, 'wb') as p:
                    pickle.dump(x, p, pickle.HIGHEST_PROTOCOL)

        del tokenizer
        del llm

    else:
        print("All constraints already embedded.")

    torch.cuda.empty_cache()
    gc.collect()
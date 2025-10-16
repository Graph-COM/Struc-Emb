import torch
import gc
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
import argparse
import os
import pickle


def chunk_list(lst, n):
    """
    Split lst into consecutive chunks of size n.
    The last chunk may be shorter if len(lst) % n != 0.
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def free_non_model_tensors(model):
    # 1) IDs of tensors you want to keep (model params + buffers)
    keep = {id(p) for p in model.parameters()}
    keep |= {id(b) for b in model.buffers()}

    live_cuda_tensors = []
    for o in gc.get_objects():
        try:
            # Some gc objects are weakref proxies; even isinstance() can raise
            if isinstance(o, torch.Tensor):
                t = o
            else:
                # Skip anything that's not a Tensor (avoid touching .data if possible)
                continue

            # Now safe to check device (guard anyway)
            if t.is_cuda:
                live_cuda_tensors.append(t)
        except ReferenceError:
            # Referent vanished; ignore
            continue
        except Exception:
            # Be defensive against any odd object types
            continue

    # 2) Free the CUDA tensors that aren't model-owned
    to_free = [t for t in live_cuda_tensors if id(t) not in keep]

    # Best you can do is drop references; if something else still holds them,
    # GC won't reclaim. Also detach to sever graph references.
    # for t in to_free:
    #     try:
    #         if t.grad_fn is not None or t.requires_grad:
    #             t = t.detach()
    #         # Drop any views to encourage freeing storage
    #         del t
    #     except Exception:
    #         pass
    
    for t in to_free:
        # detach so you don’t keep a grad‐graph ref
        cpu_copy = t.detach().cpu()
        # reassign the tensor’s data pointer to the small CPU copy
        t.data = cpu_copy.data
        # drop your handle to the CPU copy
        del cpu_copy

    # 3) Collect & release CUDA caching allocator
    gc.collect()
    torch.cuda.empty_cache()

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def encode_and_save(passages,
                    device_ids, N, D, filename, idx_start, size):
    # 1. setup
    # device_ids = [0, 1, 3, 5]
    
    # device = torch.device('cuda:1')
    # CUDA_VISIBLE_DEVICES=1,2,3,4,5
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    # model.eval()
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.to(f"cuda:{device_ids[0]}")

    tokenizer = AutoTokenizer.from_pretrained(f'Qwen/Qwen3-Embedding-{size}B', padding_side='left')
    device = torch.device(f"cuda:{device_ids[0]}")
    model = AutoModel.from_pretrained(f'Qwen/Qwen3-Embedding-{size}B').to(device)
    model = torch.nn.DataParallel(model, device_ids = device_ids)
    # for module_key, module in model._modules.items():
    #         model._modules[module_key] = _CustomDataParallel(module, device_ids=device_ids)
    
    # N = 30000          # total size
      # if wrapped in DataParallel
    emb_file = np.memmap(
        f"{filename}_embedding_{idx_start}.npy",
        dtype="float32",
        mode="w+",
        shape=(N, D)
    )

    # 2) Stream batches, write them in
    idx = idx_start
    with torch.no_grad():
        for batch in passages:
            print(idx)
            # move input to GPU
            # passage_embeddings = model.encode(batch)
            input_ids = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length = 8192).to(device)
            # passage_embeddings = model.encode(batch, instruction='', max_length=max_length).cpu().numpy()
            # B = passage_embeddings.shape[0]
            passage_embeddings = model(**input_ids)

            embeds = last_token_pool(passage_embeddings.last_hidden_state, input_ids['attention_mask'])
            embeds = embeds[:, :D]
            embeds  = F.normalize(embeds, p=2, dim=1)
            B = embeds.shape[0]
            emb_file[idx: idx + B] = embeds.cpu().numpy()
            idx += B

            # free GPU memory
            free_non_model_tensors(model)
            del batch, passage_embeddings, embeds
            torch.cuda.empty_cache()
            # all_embs.append(passage_embeddings)
            # torch.cuda.empty_cache()

    # concatenate  and save once at the end
    emb_file.flush()
    print(f"Wrote {idx}×{D} embeddings to embeddings.npy")

def get_output_file(input_file: str) -> str:
    # Remove the "_texts.pkl" suffix
    if input_file.endswith("_texts.pkl"):
        return input_file[:-10]  # length of "_texts.pkl" is 10
    else:
        # fallback: remove extension
        return os.path.splitext(input_file)[0]


def main():
    torch.set_num_threads(10)
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--model_size', type=float, default=8,
                        help='which size of model')
    parser.add_argument('--batch', type=int, default=6,
                        help='batch size')
    parser.add_argument('--devices', type=str, default='0',
                help='which gpus')
    parser.add_argument('--input_file', type=str, default='',
                        help='input file of the texts to embed')
    parser.add_argument('--start_idx', type=int, default=0,
                    help='start idx of the texts')
    
    
    args = parser.parse_args()
    print(args)

    with open(args.input_file, "rb") as f:
        passages = pickle.load(f)

    dataset = args.input_file.split("/")[0]
    print(dataset)
    
    if 'query' in args.input_file:
        if dataset == 'musique' or dataset == 'hotpot_qa':
            print('web query')
            task = 'Given a web search query, retrieve relevant passages that answer the query'
        elif dataset == 'stark':
            print('product query')
            task = 'Given a user query for product recommendation, retrieve relevant products that satisfy the query.'
        passages = [get_detailed_instruct(task, i) for i in passages]
        
    batch_size = args.batch
    passage_chunks = chunk_list(passages, batch_size)
    device_ids = list(map(int, args.devices.split("_")))
    print(device_ids)
    # device_ids = [5, 2]
    size = args.model_size
    if size == 4:
        dim = 2560
        size = 4
    elif size == 8:
        dim = 4096
        size = 8
    else:
        dim = 1024

    start_idx = args.start_idx
    if np.mod(args.start_idx, batch_size) == 0:
        start_idx_passage = int(args.start_idx / batch_size)
    else:
        start_idx_passage = int(args.start_idx / batch_size)
        start_idx = start_idx_passage * batch_size

    output_file = get_output_file(args.input_file)
    if 'query' in args.input_file:
        output_file = output_file + "_task_instruct"
    print(output_file)
    print(len(passage_chunks[0]))
    print(start_idx_passage)
    encode_and_save(passage_chunks[start_idx_passage:], device_ids, len(passages), dim, f"{output_file}_{size}B", start_idx, size)

if __name__ == "__main__":
    main()

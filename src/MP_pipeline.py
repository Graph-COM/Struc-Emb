import torch
import torch.nn.functional as F
import argparse
import os
import pickle
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import numpy as np
from ba_mp import gapemp_graph_batch_qwen, gapemp_graph_qwen_context_agg, gapemp_graph_qwen_context_agg_new, gapemp_graph_qwen_context_agg_fixed

    
def get_output_file(input_file: str) -> str:
    # Remove the "_texts.pkl" suffix
    if input_file.endswith("_concat_dict.pkl"):
        return input_file[:-16]  # length of "_texts.pkl" is 10
    elif input_file.endswith("_concat_dict_addself.pkl"):
        return input_file[:-22]
    elif input_file.endswith("concat_dict.pkl"):
        return input_file[:-15]
    elif input_file.endswith("concat_dict_addself.pkl"):
        return input_file[:-21]
    else:
        # fallback: remove extension
        return os.path.splitext(input_file)[0]

def main():
    torch.set_num_threads(10)
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--model_size', type=float, default=8,
                        help='which size of model')
    parser.add_argument('--mp_type', type=str, default="mp",
                        help='which type of message passing in encoding we use')
    parser.add_argument('--device', type=int, default=1,
                    help='gpu device')
    parser.add_argument('--input_file', type=str, default='',
                        help='input file of the texts to embed')
    parser.add_argument('--start_idx', type=int, default=0,
                    help='start idx of the texts')
    parser.add_argument('--cap', type=int, default=55000,
                help='cap of total token')
    
    args = parser.parse_args()
    print(args)

    # batch_size = args.batch
    device_ids = [5, 0, 2]
    size = args.model_size
    if size == 4:
        dim = 2560
        size = 4
    elif size == 8:
        dim = 4096
        size = 8
    else:
        dim = 1024

    # if np.mod(args.start_idx, batch_size) == 0:
    #     start_idx = args.start_idx
    #     start_idx_passage = int(args.start_idx / batch_size)
    # else:
    #     start_idx_passage = int(args.start_idx / batch_size)
    #     start_idx = start_idx * batch_size

    output_file = get_output_file(args.input_file)
    print(output_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    # max_length = 8192
    device_ids = [args.device]
    device = torch.device(f"cuda:{device_ids[0]}")
    tokenizer = AutoTokenizer.from_pretrained(f'Qwen/Qwen3-Embedding-{size}B', padding_side='left')
    model = AutoModel.from_pretrained(f'Qwen/Qwen3-Embedding-{size}B').to(device)
    # model = torch.nn.DataParallel(model, device_ids = device_ids)
    model.eval()
    # for module_key, module in model._modules.items():
    #         model._modules[module_key] = _CustomDataParallel(module, device_ids=device_ids)
    # wrap in DataParallel so that a batch of size 4*per_gpu_batch
    # is split across 4 GPUs ⇒ each GPU sees per_gpu_batch
    # model = torch.nn.DataParallel(model, device_ids=device_ids).to("cuda")
    # for module_key, module in model._modules.items():
    #     model._modules[module_key] = DataParallel(module, device_ids=device_ids).to("cuda")
    model_config = AutoConfig.from_pretrained(f"Qwen/Qwen3-Embedding-{size}B")
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=model_config).to(device=device, dtype=torch.float32)
    emb.eval()

    dataset = args.input_file.split("/")[0]
    print(dataset)

    center_node_list = []
    neighbor_nodes_list = []
    with open(args.input_file, "rb") as f:
        doc_dict = pickle.load(f)

    for doc in doc_dict:
        center_node_list.append(doc['0'])
        neighbor_nodes_list.append(doc['1'])
    
    n = len(center_node_list)

    start_idx = args.start_idx
    # temperature = 1
    scale = 1
    cap = args.cap
    max_token = 8192
    print(f"num_samples:{n}; dim:{dim}")
    if "context" in args.mp_type:
        if dataset in ['musique', 'hotpot_qa']:
            context_instruct = "Summarize the above linked Wikipedia paragraphs of the target paragraph into a contextual representation that captures shared entities, relations, and background knowledge. Use this distilled context, together with the original paragraphs as supporting evidence, when encoding the following target paragraph for retrieval: "
        elif dataset == 'stack_exchange':
            context_instruct = "Summarize the above related StackExchange post titles of the target post into a contextual representation that captures overlapping topics, recurring tags, and shared problem domains. Use this distilled context, together with the original posts as supporting evidence, when encoding the following target post for clustering: "
        elif dataset in ['cora', 'citeseer', 'pubmed']:
            context_instruct = "Summarize the above citing and cited research papers of the target paper into a contextual representation that captures shared domains, recurring methods, and notable overlaps. Use this distilled context, together with the original papers as supporting evidence, when encoding the following target paper for domain classification: "
        elif dataset == 'bookhis':
            context_instruct = "Summarize the above co-purchased or co-viewed history books of the target book into a contextual representation that highlights dominant geographical regions, historical periods, and recurring themes. Use this distilled context, together with the original books as supporting evidence, when encoding the following target book for category classification: "
        elif dataset == 'sportsfit':
            context_instruct = "Summarize the above co-purchased or co-viewed sports & fitness items of the target item into a contextual representation that captures activity types, training goals, and usage contexts. Use this distilled context, together with the original items as supporting evidence, when encoding the following target item for category classification: "
        elif dataset == 'stark':
            context_instruct = "Summarize the above co-purchased or co-viewed products, brands, colors, and categories of the target product into a contextual representation that captures complementary functions, styles, and usage contexts. Use this distilled context, together with the original attributes as supporting evidence, when encoding the following target product for recommendation: "
        center_node_list = [context_instruct + i for i in center_node_list]
        print(center_node_list[0])
        gapemp_graph_batch_qwen(tokenizer, model, emb, scale, cap, center_node_list[start_idx:], neighbor_nodes_list[start_idx:], n, dim, max_token, device, f"{output_file}_instruct_{args.mp_type}_{size}B", start_idx)
    else:
        gapemp_graph_batch_qwen(tokenizer, model, emb, scale, cap, center_node_list[start_idx:], neighbor_nodes_list[start_idx:], n, dim, max_token, device, f"{output_file}_{args.mp_type}_{size}B", start_idx)

if __name__ == "__main__":
    main()

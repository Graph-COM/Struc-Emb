<h1 align="center">Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings (Struc-EMB)</h1>

This repository is the implementation for the paper [Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings](https://arxiv.org/pdf/2510.08774).

## File Overview ##
- MP_pipeline.py: contain main function to run Struc-Emb-Par and Struc-Emb-Par-Distill
- individual_encode.py: contain main function to run individual embeddings and Struc-Emb-Seq
- ba_mp.py: contain the implementation of Struc-Emb-Par variants
- requirements.txt: the required libaries
- evaluation_citation.ipynb: the evaluation pipeline for citation network classification
- evaluation_musique.ipynb: the evaluation pipeline for MuSiQue dataset retrieval performance

## Dataset ##
In side each dataset folder, we have files:
- pkl file end with concat_dict.pkl: The processed file containing target segments and its related segments
Note:
- you can obtain the target segments text by refering to each dict['0'] and related texts by refering to each dict['1']
- you can obtain the concat texts to run Struc-Emb-Seq by concatenating the related documents with target documents

For cora:
- additionally have other files needed for evaluation, like labels, and 4o prediction on random sample to generate class embeddings
For Musique:
- query_texts.pkl: query texts file
- rel_paragraph.pkl: answers for the queries
- musique_paragraphs_concat_idx.pkl: the file that match idx need in evaluation

## Training ##
You can specify model_size to select from Qwen3 embedding 0.6/4/8B
To run Struc-Emb-Par:

`python MP_pipeline.py --model_size 0.6 --mp_type mp --device [gpu] --input_file [xxx_concat_dict.pkl] --start_idx 0`

To run Struc-Emb-Par-Distill:

`python MP_pipeline.py --model_size 0.6 --mp_type mp_context --device [gpu] --input_file [xxx_concat_dict.pkl] --start_idx 0`

To run Struc-Emb-Seq or individual embedding or post-hoc aggregation:
First process the target texts or concatenation texts, then run

`python individual_encode.py --model_size 0.6 --batch 2 --input_file [input_file_path] --start_idx 0 --devices [gpu]`




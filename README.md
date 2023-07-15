# EFO<sub>k</sub>-CQA: Towards Knowledge Graph Complex Query Answering beyond Set Operation

This repository is for implementation for the paper "EFO<sub>k</sub>-CQA: Towards Knowledge Graph Complex Query Answering beyond Set Operation".



## 1 Preparation

### 1.1 Environment

We have utilized a CSP solver provided in the python-constraint package, please install it by:
```
pip install python-constraint
```

We have also utilized the pytorch-geometric and networkx package, please install it by:
```
conda install pyg -c pyg
conda install networkx
```


### 1.1 Data Preparation


Please download the EFO<sub>k</sub>-CQA dataset from [here](https://drive.google.com/drive/folders/1kqnRdpcnVdBfbY8eVRIoVUgXdgkU8qd4?usp=sharing), 
the data of three knowledge graphs can be downloaded separately and put it in the `data` folder, as well as a file named 
`DNF_EFO2_23_4123166.csv` which is used to store the abstract query graph(query type) for the EFOX experiment.

The `DNF_EFO2_23_4123166.csv` should be put into the `data` folder.

Then, after unzipping the query data. an example data folder should look like this:
```
data/FB15k-237-EFOX-final/
  - kgindex.json
  - train_kg.tsv
  - valid_kg.tsv
  - test_kg.tsv
  - test-type0000-EFOX-qaa.json 
  - test-type0001-EFOX-qaa.json
  - ......
```

where the `test-type0000-EFOX-qaa.json` is used for the EFOX experiment, containing the data for query type0000. 

The `kgindex.json` and `train_kg.tsv` are the index file and the training graph for the knowledge graph respectively, 
the `valid_kg.tsv` and `test_kg.tsv` are the validation graph and the test graph respectively. They are used for data generation.

In each `test-type0000-EFOX-qaa.json`, it contains a dict with the key be the formula and the value is a list of query:
{formula: [query1, query2, ...]}.




### 1.2 Checkpoint Preparation

To reproduce the experiment in the paper, we have provided the checkpoint for each model foreach knowledge graph, we
offer the checkpoint for six representative model (BetaE, LogicE, ConE, CQD, LMPNN, FIT), which can be downloaded from [here](https://drive.google.com/drive/folders/13S3wpcsZ9t02aOgA11Qd8lvO0JGGENZ2?usp=sharing),


It should be unzipped and put in the `ckpt` folder.

An example of the `ckpt` sub folder, which includes the model trained on the knowledge graph ``FB15k-237'' should look like this:
```
ckpt/FB15k-237
  - BetaE_full/checkpoint
  - LogicE_full/450000.ckpt
  - ConE_full/300000.ckpt
  - CQD/FB15k-237-model-rank-1000-epoch-100-1602508358.pt
  - LMPNN/lmpnn-FB15k-237.ckpt
  - FIT/torch_0.005_0.001.ckpt
```

where each sub folder is the checkpoint for each model, and the name of the sub folder is the name of the model.

## 2. Sample the data yourself

We have the powerful frame that supports several key functionalities for the task of complex query answering, 
you can also sample the query by yourself following the instruction. 

If you have downloaded the EFO<sub>k</sub>-CQA dataset, you can also skip this section.



### 2.1 Enumerate the abstract query graph
To try to enumerate the abstract query graph, please run the following command:
```angular2html
python data_preparation/create_qg.py
```
it should output the abstract query graph in the `data` folder, with the name of `DNF_EFO2_23_4123166.csv_filtered.csv`, 
which is used to store the abstract query graph(query type) for the EFOX experiment. You can also change hyperparameters used 
in this code to explore other possible combinatorial space different from the one in the paper.


### 2.2 Sample the query graph
Ground the abstract query graph to become a query graph requires two functionalities: 
1. Ground entities and relations, 
2. Compute the answer for the grounded query graph.

We give a example of sampling the query graph for the knowledge graph FB15k-237, where each type of query has 
1000 samples if it does not negation and 500 otherwise, please run the following command:

```angular2html
python data_preparation/sample_query.py --output_folder data/FB15k-237-EFOX-final --data_folder data/FB15k-237-EFOX-final --num_positive 1000 --num_negative 500
```

## 3. Reproduce the result of the paper.

### 3.1 Query embedding method
For query embedding method, including BetaE, LogicE, ConE, please run the following command:

```angular2html
python QG_EFOX.py --config config/LogicE_FB15k-237_EFOX.yaml
```

which is an example for LogicE method on FB15k-237 dataset. The config file in the `config` folder is used to specify 
the model and knowledge graph used in the experiment.

### 3.2 Query graph method: CQD + LMPNN

For LMPNN, note that you need to download the CQD checkpoint as well checkpoint of LMPNN since LMPNN is built upon CQD,
please run the following command for LMPNN on FB15k-237 dataset:

```angular2html
python evaluate_lmpnn.py \
  --task_folder data/FB15k-237-EFOX-final \
  --checkpoint_path ckpt/FB15k-237/CQD/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --checkpoint_path_lmpnn ckpt/FB15k-237/LMPNN/lmpnn-FB15K-237.ckpt \
  --embedding_dim 1000
```


For LMPNN on FB15k:
```angular2html
python evaluate_lmpnn.py \
  --task_folder data/FB15k-EFOX-final \
  --checkpoint_path ckpt/FB15k/CQD/FB15k-model-rank-1000-epoch-100-1602520745.pt \
  --checkpoint_path_lmpnn ckpt/FB15k/LMPNN/lmpnn-FB15K.ckpt \
  --hidden_dim 8192 
```


For LMPNN on NELL:

```angular2html
python3 evaluate_lmpnn.py \
  --task_folder data/NELL-EFOX-final \
  --checkpoint_path ckpt/NELL/CQD/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --checkpoint_path_lmpnn ckpt/NELL/LMPNN/lmpnn-NELL.ckpt \
  --hidden_dim 8192 \
  --temp 0.1 
```

For CQD on FB15k-237:
```angular2html
python evaluate_lmpnn.py \
  --task_folder data/FB15k-237-EFOX-final \
  --reasoner gradient \
  --checkpoint_path ckpt/FB15k-237/CQD/FB15k-237-model-rank-1000-epoch-100-1602508358.pt 
```

For CQD on FB15k:

```angular2html
python evaluate_lmpnn.py \
  --task_folder data/FB15k-EFOX-final \
  --reasoner gradient \
  --checkpoint_path ckpt/FB15k/CQD/FB15k-model-rank-1000-epoch-100-1602520745.pt \
  --hidden_dim 8192 
```

For CQD on NELL:

```angular2html
python3 evaluate_lmpnn.py \
  --task_folder data/NELL-EFOX-final \
  --reasoner gradient \
  --checkpoint_path ckpt/NELL/CQD/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --hidden_dim 8192 
```
### 3.3 Query graph method: FIT

For FIT, please run the following command to run the expriment on FB15k-237: 

```angular2html
python solve_EFOX.py 
```
 
If you want to try to use FIT on KB15k or NELL, please run the following command:

```angular2html
python solve_EFOX.py  --ckpt ckpt/FB15k/FIT/torch_0.005_0.001.ckpt --data_folder data/FB15k-EFOX-final
```

```angular2html
python solve_EFOX.py  --ckpt ckpt/NELL/FIT/torch_0.0002_0.001.ckpt --data_folder data/NELL-EFOX-final
```


We note it may encounter out-of-memory error when running the FIT model on KB15k and NELL as mentioned in the paper, 
which indicates that FIT face the challenge of scalability.

## 4. Aggregate the final result.

As there are numerous abstract query graphs (query types), we can aggregate the result of each query type to get the 
final result.

Please create a folder to record the benchmark result, just like the following:
```
result/FB15k-237_result
-BetaE_test
-LogicE_test
-ConE_test
-CQD_test
-LMPNN_test
-FIT_test
```

Then run the following code and get the presentation of the table, for knowledge graph FB15k-237, queries with one free variable: 
```angular2html
python construct_and_analyze.py --out_folder result --dataset FB15k-237 --model LogicE --variable 1 --construct 1
```

The `--variable` is used to specify the number of variable in the query graph, it can be set to 1 or 2.
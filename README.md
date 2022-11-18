# SR-GAT
## Origin data
We use the cross-sentence n-ary relation extraction dataset provided by Peng et al. It is a biomedical dataset from the PubMed literature database that contains three types of entities: "drug", "gene", and "mutation", including 6987 instances of "drug-gene-mutation" ternary relationships and 6087 instances of "mutation-drug" binary relationships. You can get the original dataset here: https://github.com/freesunshine0316/nary-grn.
## Preprocess data
Please modify the following parameters in the `preprocessing.py` file according to the original data storage path.
``` 
self.data_path_list = ['origin_data/drug_gene_var/0/data_graph_1', 'origin_data/drug_gene_var/0/data_graph_2',
                          'origin_data/drug_gene_var/1/data_graph_1', 'origin_data/drug_gene_var/1/data_graph_2',
                          'origin_data/drug_gene_var/2/data_graph_1', 'origin_data/drug_gene_var/2/data_graph_2',
                          'origin_data/drug_gene_var/3/data_graph_1', 'origin_data/drug_gene_var/3/data_graph_2',
                          'origin_data/drug_gene_var/4/data_graph_1', 'origin_data/drug_gene_var/4/data_graph_2']

self.vec_path = 'vec/vocab_morph.wordvec.st'
```
After the program is executed, the following data will be stored in the `data/` folder.
```
--edge_id2words.json
--edge_word_vec.npy
--edge_word2ids.json
--id2words.json
--word2ids.json
--wordvec.npy
--data(1)
  --test_entity_index.npy
  --test_label_ids.npy
  --test_sentence_ids.npy
  --test_y.npy
  --train_entity_index.npy
  --train_label_ids.npy
  --train_sentence_ids.npy
  --train_y.npy
```
## Model Training
Set class_num (2 or 5, class_num indicates the number of relational categories, i.e., two classification and five classification), execute `python run.py`
```
training 0 epoch...
  1%|▏         | 72/5219 [00:23<28:04,  3.06it/s]
```
## Result
```
test set macro accuracy: 0.9133435582822086
test set macro precision: 0.8902972279845123
test set macro recall: 0.9084080450541909
test set macro f1: 0.8977559548819014
```

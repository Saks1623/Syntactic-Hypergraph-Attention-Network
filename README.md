# Document Classification via Hypergraph Attention Networks (HyperGAT)

## Overview
This repository provides a PyTorch implementation for a document classification model that utilizes Hypergraph Attention Networks (HyperGAT). The model constructs a hypergraph for each document, capturing complex structural, semantic, and syntactic relationships between words.

It supports multiple types of hyperedges to enrich the document representation:
1. **Sentence Hyperedges:** Connect words occurring in the same sentence.
2. **Semantic Hyperedges (LDA):** Connect words belonging to the same latent topics generated via Latent Dirichlet Allocation (LDA).
3. **Syntactic Hyperedges:** Connect head words and their dependents using dependency parsing (via SpaCy).

The model can be initialized using pre-trained **GloVe embeddings** or **BERT embeddings** (including domain-specific fine-tuned BERT models).

## Project Structure

* `run.py`: The main entry point to parse arguments, load data, and execute the training and evaluation loops.
* `model.py`: Contains the model architecture, including `DocumentGraph` and `HGNN_ATT`, as well as the training and testing procedures.
* `layers.py`: Implementation of the sparse `HyperGraphAttentionLayerSparse`, managing the node-to-edge and edge-to-node message passing operations.
* `preprocess.py`: Handles data loading, tokenization, vocabulary building, and embedding matrix initialization (GloVe/BERT).
* `utils.py`: Utility functions for text cleaning, dataset splitting, and the `Data` object which handles batching and dynamic sparse matrix construction.
* `fine_tuned_bert.py`: A standalone script to fine-tune a BERT language model (`bert-base-uncased`) on the target dataset corpus using Masked Language Modeling (MLM).
* `generate_lda.py`: Generates semantic topics using Scikit-Learn's LDA and maps top keywords for semantic hyperedge construction.
* `generate_syntectic.py`: Extracts syntactic hyperedges using SpaCy's dependency parser based on token head-child relationships.

## Requirements

Ensure you have Python 3.7+ installed. Install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn nltk spacy networkx scipy pandas tqdm
```

### Additionally, download the necessary NLTK and SpaCy resources:
```bash
python -m nltk.downloader stopwords wordnet punkt
python -m spacy download en_core_web_sm
```

## Data Preparation

Place your dataset files in a `data/` directory. The expected format is:

* `data/<dataset>_corpus.txt`: The raw text documents, one document per line.
* `data/<dataset>_labels.txt`: The corresponding labels and train/test splits. Expected format per line: `<doc_id> \t <train/test> \t <label>`

If you are using GloVe embeddings (the default for certain datasets like `mr` without the `--use_bert` flag), download `glove.6B.300d.txt` and place it in the `data/` directory.

## Usage

### 1. (Optional) Fine-Tune BERT
To use domain-adapted BERT embeddings, fine-tune BERT on your corpus first:

```bash
python fine_tuned_bert.py --dataset <dataset_name>
```

### 2. (Optional) Generate LDA Semantic Hyperedges
To utilize semantic hyperedges (`--use_LDA`), generate the LDA topics prior to training:

```bash
python generate_lda.py --dataset <dataset_name> --topics 6 --topn 10
```

### 3. Train and Evaluate the Model
Run the main script to start training. You can toggle syntactic edges, LDA edges, and BERT embeddings using command-line flags.

```bash
python run.py --dataset <dataset_name>
Run with Syntactic Edges and BERT Embeddings:
```
```Bash
python run.py --dataset <dataset_name> --use_syn --use_bert
Run with all features (Syntactic + LDA + BERT):

```
```Bash
python run.py --dataset <dataset_name> --use_syn --use_LDA --use_bert --epoch 20
```
### Key Command Line Arguments (run.py)
--dataset: Name of the dataset (e.g., R52, 20ng, mr). Default: R52.

--batchSize: Batch size for training. Default: 8.

--hiddenSize: Dimension of the hidden state. Default: 100.

--initialFeatureSize: Initial feature dimension (e.g., 768 for BERT, 300 for GloVe). Default: 768.

--epoch: Number of training epochs. Default: 10.

--lr: Learning rate. Default: 0.001.

--dropout: Dropout probability. Default: 0.3.

--use_LDA: Flag to include LDA-based semantic hyperedges.

--use_syn: Flag to include dependency parsing-based syntactic hyperedges.

--use_bert: Flag to use BERT embeddings instead of random or GloVe initializations.

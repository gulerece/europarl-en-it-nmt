# Neural Machine Translation with RNNs and Attention

This project investigates neural machine translation using GRU-based sequence-to-sequence models. The experiments focus primarily on English–Italian translation and compare different word-embedding strategies, character-level modeling, attention, and pivot translation.

## Project Overview

The notebook covers:

- Exploration and preprocessing of the Europarl parallel corpus
- English → Italian and Italian → English translation
- Comparison of Word2Vec, GloVe, FastText, and random embeddings
- Word-level and character-level models
- Hyperparameter experiments
- An attention-based sequence-to-sequence model
- Italian → English → Swedish pivot translation
- Evaluation using loss, token accuracy, macro F1, and BLEU

## Repository Contents

| File | Description |
|---|---|
| `europarl-en-it-nmt.ipynb` | Complete implementation and experiments |
| `report.pdf` | Project report and discussion |
| `ExperimentSummarySheet.xlsx` | Summary of experimental results |

## Data

The experiments use:

- Europarl Italian–English parallel corpus
- Europarl Swedish–English parallel corpus
- GloVe 6B 100-dimensional English embeddings

The datasets and trained model files are not included in this repository due to their size.

Expected Google Drive structure:

```text
MyDrive/
└── NLP_Project_2.2/
    └── data/
        ├── europarl-v7.it-en.en
        ├── europarl-v7.it-en.it
        ├── europarl-v7.sv-en.en
        ├── europarl-v7.sv-en.sv
        └── glove.6B.100d.txt
```

The data location can be changed in the notebook:

```python
DATA_DIR = "/content/drive/MyDrive/NLP_Project_2.2/data"
```

## Running the Notebook

The notebook was designed to run in Google Colab.

1. Upload the required datasets to Google Drive using the structure above.
2. Open `europarl-en-it-nmt.ipynb` in Google Colab.
3. Select a GPU runtime: `Runtime → Change runtime type → GPU`.
4. Run the notebook cells sequentially.

Because the notebook trains several models for 50 epochs, running the complete notebook may take considerable time and GPU memory.

## Dependencies

The main Python libraries are:

- PyTorch
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- NLTK
- Gensim
- tqdm

Gensim is installed directly inside the notebook. NLTK tokenization resources are downloaded automatically.

## Preprocessing

The preprocessing pipeline:

1. Removes empty sentence pairs and XML metadata
2. Filters sentences longer than 80 tokens
3. Randomly samples 10% of the corpus
4. Converts text to lowercase
5. Removes digits and punctuation
6. Tokenizes the sentences
7. Splits the data into training, validation, and test sets

A fixed random seed is used where applicable to improve reproducibility.

## Model Architecture

The baseline translation system uses:

- A trainable source embedding layer
- A GRU encoder
- A GRU decoder
- Teacher forcing during training
- Greedy autoregressive decoding during inference

The main configuration uses:

| Parameter | Value |
|---|---:|
| Embedding dimension | 100 |
| Hidden dimension | 128 |
| Maximum vocabulary size | 30,000 |
| Maximum sentence length | 80 |
| Batch size | 512 |
| Epochs | 50 |
| Learning rate | 0.001 |

## Main Results

### English → Italian

| Embedding | Accuracy | Macro F1 | BLEU |
|---|---:|---:|---:|
| Word2Vec | 0.2382 | 0.0382 | 0.0203 |
| GloVe | 0.2265 | 0.0355 | 0.0000 |
| FastText | 0.2487 | 0.0407 | 0.0127 |
| Random | 0.2348 | 0.0377 | 0.0198 |

### Italian → English

| Embedding | Accuracy | Macro F1 | BLEU |
|---|---:|---:|---:|
| Word2Vec | 0.2988 | 0.0400 | 0.0231 |
| FastText | 0.3080 | 0.0370 | 0.0187 |
| Random | 0.2960 | 0.0346 | 0.0302 |

*GloVe is excluded here because the embeddings used are English-only.*

### Additional Experiments

| Experiment | Accuracy | Macro F1 | BLEU |
|---|---:|---:|---:|
| Character-level EN → IT | 0.6404 | 0.1927 | — |
| Word2Vec with attention | 0.3894 | 0.1428 | 0.0489 |
| English → Swedish | 0.2815 | 0.0434 | 0.0248 |

*BLEU was not computed for the character-level experiment because the reported BLEU evaluation was designed for word/token n-gram overlap rather than character-level outputs.*

The attention model produced the strongest English–Italian result and improved BLEU substantially over the baseline models.

## Notes and Limitations

- The notebook depends on Google Drive paths and is not directly portable without changing `DATA_DIR`.
- Trained model checkpoints are saved to Google Drive and are not included here.
- BLEU scores remain low, showing the limitations of the relatively simple GRU architecture and greedy decoding strategy.
- The English → Italian GloVe model's BLEU score of `0.0000` is the recorded evaluation result, not a missing value. Its generated translations had insufficient word-level n-gram overlap with the references, which can occur with short sequences, rare words, and weak lexical alignment despite some token-level accuracy.
- Token accuracy can be misleading for translation because frequent words may dominate the score.

## Report

For the full methodology, experimental discussion, and conclusions, see [`report.pdf`](report.pdf).

## Authors

- Ece Güler
- Benjamin Kuntze-Fechner

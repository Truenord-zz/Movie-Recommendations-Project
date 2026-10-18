# Recommender systems for the video games category on the amazon dataset

I use my recommender systems knowledge on an Amazon dataset to make several movie recommendation models.

The project runs three different kinds of methods

1) Collaborative filtering 
2) NLP and Content-based methods
3) Finally some custom recommendation frameworks

The entire dataset are compared on Amazon video game ratings (1,389 users, 932 items, 98% sparse).

## Models

| Type | Models |
|---|---|
| Baseline | TopPop |
| Collaborative filtering | Item-based KNN (cosine, implicit), SVD (SVD of the relevant matrix) |
| Graph methods | RP3beta, GF-CF, BPR-MF, LightGCN |
| Content-based | User profile = rating-weighted mean of rated items, represented with TF-IDF, GloVe, BERT or RoBERTa |
| Hybrids | Parallel (reciprocal rank fusion), switching (content-based for sparse users), sequential (content-based ratings fill in sparse users before CF) |

Ratings of 3 or higher count as relevant. KNN, SVD and the graph methods train only on ratings of 3 or higher. The settings of every model except the content-based ones and the hybrids are picked on a validation split of the training ratings (`tune_*.py`), not on the test set.

## Results

Top-10 recommendations, a test rating of 3 or higher counts as relevant.

| Model | HR | P@10 | MAP@10 | MRR@10 | nDCG@10 | Coverage |
|---|---|---|---|---|---|---|
| TopPop | 0.230 | 0.026 | 0.021 | 0.077 | 0.047 | 0.020 |
| KNN | 0.461 | 0.062 | 0.064 | 0.202 | 0.123 | 0.704 |
| SVD | 0.407 | 0.053 | 0.052 | 0.173 | 0.103 | 0.310 |
| LightGCN | 0.449 | 0.061 | 0.062 | 0.198 | 0.120 | 0.669 |
| BPR-MF | 0.329 | 0.041 | 0.036 | 0.123 | 0.076 | **0.967** |
| RP3beta | **0.478** | **0.065** | **0.069** | **0.215** | **0.131** | 0.768 |
| GF-CF | 0.468 | 0.063 | 0.068 | 0.214 | 0.128 | 0.496 |
| CB TF-IDF | 0.205 | 0.023 | 0.020 | 0.075 | 0.044 | 0.613 |
| CB GloVe | 0.092 | 0.010 | 0.009 | 0.035 | 0.020 | 0.227 |
| CB BERT | 0.084 | 0.009 | 0.006 | 0.030 | 0.015 | 0.264 |
| CB RoBERTa | 0.068 | 0.007 | 0.006 | 0.024 | 0.014 | 0.231 |
| Parallel RRF (KNN+CB) | 0.401 | 0.051 | 0.053 | 0.179 | 0.104 | 0.771 |
| Parallel RRF (SVD+CB) | 0.362 | 0.044 | 0.044 | 0.154 | 0.088 | 0.605 |
| Switching (KNN+CB) | 0.346 | 0.047 | 0.039 | 0.148 | 0.083 | 0.802 |
| Switching (SVD+CB) | 0.317 | 0.041 | 0.034 | 0.131 | 0.074 | 0.700 |
| Sequential (CB -> KNN) | 0.451 | 0.061 | 0.061 | 0.198 | 0.120 | 0.697 |
| Sequential (CB -> SVD) | 0.397 | 0.051 | 0.050 | 0.169 | 0.100 | 0.313 |

RP3beta, GF-CF, KNN and LightGCN are very close to each other in performance. And notably, none of the hybrid methods beat a simple item-based KNN.


## Data
The dataset is a sample (30,000 train and 7,500 test ratings) of the Video Games category of the Amazon Reviews 2023 dataset from McAuley Lab https://amazon-reviews-2023.github.io/.

## Structure

```
hybrid_recommenders/
    data.py                 loading and cleaning ratings and metadata
    text_preprocessing.py   boilerplate removal, lemmatization, stopwords
    collaborative.py        TopPop, KNN and SVD
    content_based.py        TF-IDF, GloVe and transformer embeddings, user profiles
    lightgcn.py             LightGCN with BPR training
    graph_methods.py        RP3beta and GF-CF
    implicit_feedback.py    shared by KNN, RP3beta and GF-CF
    hybrids.py              parallel, switching and sequential hybrids
    evaluation.py           hit rate, precision, MAP, MRR, nDCG, coverage
    plotting.py
notebooks/
    01_data_exploration.ipynb
    02_collaborative_filtering.ipynb
    03_content_based.ipynb
    04_hybrids.ipynb
run_experiments.py          trains every model and prints the table above
tune_lightgcn.py            number of LightGCN layers, on a validation split
tune_graph_methods.py       RP3beta and GF-CF settings, on a validation split
tune_svd.py                 number of SVD factors, on a validation split
```

## Running

Put `train_video_games.parquet`, `test_video_games.parquet` and `meta_video_games.parquet` in `data/`.

```
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
python run_experiments.py

```

`--skip-embeddings` runs everything except GloVe, BERT and RoBERTa. The embeddings are cached in `cache/` after the first run.

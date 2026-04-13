# NLP Sentiment Classifier

From-scratch NLP sentiment analysis on the IMDb 50K movie reviews dataset, with custom tokenization, vocabulary building, preprocessing, and a PyTorch classifier.

## What this project includes

- Data cleaning from raw CSV into `sample_reviews.txt`
- Custom `Tokenizer` (`<PAD>`, `<UNK>`, top-10k vocab, encode/decode, save/load)
- Preprocessing pipeline that pads/truncates reviews to fixed length
- Sentiment model (`Embedding -> masked mean pooling -> MLP`)
- Notebook workflows for vocab analysis, training, and post-training analysis
- Math notes for cosine similarity and log-likelihood derivations

## Main files

```
src/clean.py
src/tokenizer.py
src/prep_sentiment_data.py
src/sentiment_model.py
notebooks/vocabulary_analysis.ipynb
notebooks/train_model.ipynb
notebooks/analysis.ipynb
experiments/predictive_words.txt
experiments/attention_examples.html
math-notes/cosine_similarity.md
math-notes/log_likelihood.md
```

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas torch
python3 src/clean.py
```

## Notes

- Keep `IMDB Dataset.csv` in `data/`
- Generated artifacts include `data/vocab.json`, `data/train_dataset`, `data/validation_dataset`, and `data/test_dataset`
- Trained weights are saved under `checkpoints/`

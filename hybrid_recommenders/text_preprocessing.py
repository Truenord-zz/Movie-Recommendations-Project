from string import punctuation

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Found by counting the most common first and last sentences of the descriptions
AMAZON_BOILERPLATE_PHRASES = [
    "Product Description",
    "Product description",
    "From the Manufacturer",
    "Amazon.com",
    "view larger",
    "Additional Screenshots",
    "account required for game activation and installation",
    "account required to download and install.",
    "View larger",
    "View larger.",
    "See more",
    "Screenshots",
    "*The DLC (Downloadable Content), Trials/Subscriptions may or may not be included and are not guaranteed to work*",
    "Refurbished and Sealed, may or may not include instructions, may be greatest hits/platinum hits version",
]

ENGLISH_STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
detokenizer = TreebankWordDetokenizer()


def join_into_single_text(field_value) -> str:
    """Metadata fields are stored as arrays of sentences, so we join them into one string."""
    if isinstance(field_value, (np.ndarray, list)):
        return " ".join(field_value)
    return str(field_value)


def remove_boilerplate(text: str) -> str:
    for phrase in AMAZON_BOILERPLATE_PHRASES:
        text = text.replace(phrase, "")
    return text


def normalize_text(text: str) -> str:
    """Lowercase, lemmatize and remove stopwords and punctuation."""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    tokens = [token for token in tokens if token not in punctuation]
    return detokenizer.detokenize(tokens)


def preprocess_column(column: pd.Series) -> pd.Series:
    processed_texts = []
    for field_value in column:
        text = join_into_single_text(field_value)
        text = remove_boilerplate(text)
        text = normalize_text(text)
        processed_texts.append(text)
    return pd.Series(processed_texts, index=column.index)


def lightly_clean_column(column: pd.Series) -> pd.Series:
    """Only boilerplate removal and lowercasing. Used for BERT and RoBERTa, which have their own tokenizers."""
    cleaned_texts = []
    for field_value in column:
        text = join_into_single_text(field_value)
        text = remove_boilerplate(text)
        cleaned_texts.append(text.lower())
    return pd.Series(cleaned_texts, index=column.index)


def build_item_corpus(item_metadata_df: pd.DataFrame) -> pd.Series:
    """One document per item: description, categories, features and title."""
    descriptions = preprocess_column(item_metadata_df["description"])
    categories = preprocess_column(item_metadata_df["categories"])
    features = preprocess_column(item_metadata_df["features"])
    titles = preprocess_column(item_metadata_df["title"])

    corpus = descriptions + " " + categories + " " + features + " " + titles
    return corpus


def vocabulary_size_per_preprocessing_step(descriptions: pd.Series) -> dict[str, int]:
    """Vocabulary size after each step, used for the preprocessing table in the report."""
    texts = [remove_boilerplate(join_into_single_text(description)) for description in descriptions]

    tokenized_texts = [word_tokenize(text) for text in texts]
    vocabulary_sizes = {"raw": _vocabulary_size(tokenized_texts)}

    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    vocabulary_sizes["lowercased"] = _vocabulary_size(tokenized_texts)

    tokenized_texts = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in tokenized_texts]
    vocabulary_sizes["lemmatized"] = _vocabulary_size(tokenized_texts)

    tokenized_texts = [[token for token in tokens if token not in ENGLISH_STOPWORDS] for tokens in tokenized_texts]
    vocabulary_sizes["stopwords removed"] = _vocabulary_size(tokenized_texts)

    tokenized_texts = [[token for token in tokens if token not in punctuation] for tokens in tokenized_texts]
    vocabulary_sizes["punctuation removed"] = _vocabulary_size(tokenized_texts)

    return vocabulary_sizes


def _vocabulary_size(tokenized_texts: list[list[str]]) -> int:
    vocabulary = set()
    for tokens in tokenized_texts:
        vocabulary.update(tokens)
    return len(vocabulary)

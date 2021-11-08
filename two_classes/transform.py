from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
def transform(dataframe:pd.DataFrame, train=False):
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    texts = dataframe['texts'].values
    labels = dataframe['labels'].values
    if train:
        train_texts, val_texts, train_lavels, val_lavels = train_test_split(texts.tolist(), labels.tolist(), test_size=0.2, shuffle=True, random_state=42)
        train_encodins = tokenizer(train_texts, padding=True)
        val_encodins = tokenizer(val_texts, padding=True)
        return train_encodins, train_lavels, val_encodins, val_lavels
    else:
        encodings = tokenizer(texts.tolist(), padding=True)
        return encodings, labels.tolist()
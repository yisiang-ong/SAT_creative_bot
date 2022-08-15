from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    BertTokenizer,
    BertModel,
    AutoModelWithLMHead,
    AutoTokenizer
)
from nltk.corpus import stopwords
import pytorch_lightning as pl
import textdistance as td
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import re
import nltk
import pandas as pd
nltk.download("stopwords")


# T5:

# class T5FineTuner(pl.LightningModule):
#     def __init__(self, hparams):
#         super(T5FineTuner, self).__init__()
#         self.hparams = hparams

#         self.model = T5ForConditionalGeneration.from_pretrained(
#             hparams.model_name_or_path)
#         self.tokenizer = T5Tokenizer.from_pretrained(
#             hparams.tokenizer_name_or_path)

#     def forward(
#         self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
#     ):
#         return self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             lm_labels=lm_labels,
#         )


# args_dict = dict(
#     model_name_or_path='t5-small',
#     tokenizer_name_or_path='t5-small',
# )
# args = argparse.Namespace(**args_dict)

# RoBERTa/Bert:


@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def forward(self, input):
        return mish(input)


class ClassificationModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)

        return self.classifier(hidden_states[0][:, 0, :])


# labels for emotion classification
labels = ["sadness", "joy", "anger", "fear"]
label2int = dict(zip(labels, list(range(len(labels)))))
# initialize bert tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()
# readcsv
df = pd.read_csv("pre_compute_label.csv", encoding='UTF-8')


# # load emotion classifier (T5)
# with torch.no_grad():
#     emo_model = T5FineTuner(args)
#     emo_model.load_state_dict(torch.load(
#         'T5_emotion_2ft_2.pt', map_location=torch.device('cpu')), strict=False)


# #load emotion classifier (RoBERTa)
# with torch.no_grad():
#     emo_model = ClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
#     emo_model.load_state_dict(torch.load('RoBERTa_emotion_2ft_2.pt', map_location=torch.device('cpu')), strict=False) #change path

# load emotion classifier (BERT)
with torch.no_grad():
    emo_model = ClassificationModel(BertModel.from_pretrained(
        "bert-base-uncased").base_model, len(labels))
    emo_model.load_state_dict(torch.load(
        'BERT_emotion_1ft.pt', map_location=torch.device('cpu')), strict=False)  # change path


# def get_emotion(text): # T5
#     '''
#     Computes and returns an emotion label given an utterance
#     '''
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.lower()
#     with torch.no_grad():
#         input_ids = emo_model.tokenizer.encode(
#             text + '</s>', return_tensors='pt')
#         output = emo_model.model.generate(input_ids=input_ids, max_length=2)
#         dec = [emo_model.tokenizer.decode(
#             ids, skip_special_tokens=True) for ids in output]
#     label = dec[0]
#     return label

# def get_emotion(text): # roberta
#   '''
#   Classifies and returns the underlying emotion of a text string
#   '''
#   text = re.sub(r'[^\w\s]', '', text)
#   text = text.lower()
#   t = ByteLevelBPETokenizer(
#             "tokenizer/vocab.json", #change path
#             "tokenizer/merges.txt"  #change path
#         )
#   t._tokenizer.post_processor = BertProcessing(
#             ("</s>", t.token_to_id("</s>")),
#             ("<s>", t.token_to_id("<s>")),
#         )
#   t.enable_truncation(512)
#   t.enable_padding(pad_id=t.token_to_id("<pad>"))
#   tokenizer = t
#   encoded = tokenizer.encode(text)
#   sequence_padded = torch.tensor(encoded.ids).unsqueeze(0)
#   attention_mask_padded = torch.tensor(encoded.attention_mask).unsqueeze(0)
#   with torch.no_grad():
#       output = emo_model((sequence_padded, attention_mask_padded))
#   top_p, top_class = output.topk(1, dim=1)
#   label = int(top_class[0][0])
#   label_map = {v: k for k, v in label2int.items()}
#   return label_map[label]



def get_emotion(text):  # BERT
    '''
    Classifies and returns the underlying emotion of a text string
    '''
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    encoded = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    sequence_padded = torch.tensor(encoded["input_ids"])
    attention_mask_padded = torch.tensor(encoded["attention_mask"])
    with torch.no_grad():
        output = emo_model((sequence_padded, attention_mask_padded))
    top_p, top_class = output.topk(1, dim=1)
    label = int(top_class[0][0])
    # print(label)
    label_map = {v: k for k, v in label2int.items()}
    return label_map[label]


def get_distance(s1, s2):
    '''
    Computes a distance score between utterances calculated as the overlap
    distance between unigrams, plus the overlap distance squared over bigrams,
    plus the overlap distance cubed over trigrams, etc (up to a number of ngrams
    equal to the length of the shortest utterance)
    '''
    s1 = re.sub(r'[^\w\s]', '', s1.lower())  # preprocess
    s2 = re.sub(r'[^\w\s]', '', s2.lower())
    s1_ws = regextokenizer.tokenize(s1)  # tokenize to count tokens later
    s2_ws = regextokenizer.tokenize(s2)
    # the max number of bigrams is the number of tokens in the shorter sentence
    max_n = len(s1_ws) if len(s1_ws) < len(s2_ws) else len(s2_ws)
    ngram_scores = []
    for i in range(1, max_n+1):
        s1grams = nltk.ngrams(s1.split(), i)
        s2grams = nltk.ngrams(s2.split(), i)
        # we normalize the distance score to be a value between 0 and 10, before raising to i
        ngram_scores.append(
            (td.overlap.normalized_distance(s1grams, s2grams))**i)
    normalised_dis = sum(ngram_scores)/(max_n)  # normalised
    return normalised_dis


def compute_distances(sentence, dataframe):
    '''
    Computes a list of distances score between an utterance and all the utterances in a dataframe
    '''
    distances = []
    for index, row in dataframe.iterrows():
        # assuming the dataframe column is called 'sentences'
        df_s = dataframe['sentences'][index]
        distance = get_distance(df_s.lower(), sentence)
        distances.append(distance)
    return distances


def novelty_score(sentence, dataframe):
    '''
    Computes the mean of the distances beween an utterance
    and each of the utterances in a given dataframe
    '''
    if dataframe.empty:
        score = 1.0
    else:
        d_list = compute_distances(sentence, dataframe)
        d_score = sum(d_list)
        score = d_score / len(d_list)
    return round(score, 2)

    
    
def get_sentence_score(sentence, dataframe):
    '''
    Calculates how fit a sentence is based on its weighted empathy, fluency
    and novelty values
    '''
    tmp_df = df[df["Response"]==sentence]
    empathy = float(tmp_df.empathy_score)
    fluency = float(tmp_df.fluency_score)
    novelty = novelty_score(sentence, dataframe)
    sentiment = float(tmp_df.sentiment_score)
    score = 1.5*empathy + fluency + 2*novelty +0.5*sentiment
    return score

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
nltk.download("stopwords")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math



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

# initialize bert tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# labels for emotion classification
labels = ["sadness", "joy", "anger", "fear"]
label2int = dict(zip(labels, list(range(len(labels)))))


# # load emotion classifier (T5)
# with torch.no_grad():
#     emo_model = T5FineTuner(args)
#     emo_model.load_state_dict(torch.load(
#         'T5_emotion_2ft_2.pt', map_location=torch.device('cpu')), strict=False)


# #load emotion classifier (RoBERTa)
with torch.no_grad():
    emo_model = ClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
    emo_model.load_state_dict(torch.load('RoBERTa_emotion_2ft_2.pt', map_location=torch.device('cpu')), strict=False) #change path

# load emotion classifier (BERT)
# with torch.no_grad():
#     emo_model = ClassificationModel(BertModel.from_pretrained(
#         "bert-base-uncased").base_model, len(labels))
#     emo_model.load_state_dict(torch.load(
#         'BERT_emotion_1ft.pt', map_location=torch.device('cpu')), strict=False)  # change path


# # load empathy classifier (T5)
# with torch.no_grad():
#     emp_model = T5FineTuner(args)
#     emp_model.load_state_dict(torch.load(
#         'T5_empathy_2ft_2.pt', map_location=torch.device('cpu')), strict=False)  # change path

# #load empathy classifier (RoBERTa)
with torch.no_grad():
    emp_model = ClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, 2)
    emp_model.load_state_dict(torch.load('RoBERTa_empathy_2ft_2.pt', map_location=torch.device('cpu')), strict=False) #change path

# load empathy classifier (BERT)
# with torch.no_grad():
#     emp_model = ClassificationModel(BertModel.from_pretrained(
#         "bert-base-uncased").base_model, 2)
#     emp_model.load_state_dict(torch.load(
#         'BERT_empathy_1ft_1.pt', map_location=torch.device('cpu')), strict=False)  # change path


# Load pre-trained GPT2 language model weights
with torch.no_grad():
    gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    gptmodel.eval()

# Load pre-trained GPT2 tokenizer
gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()


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

def get_emotion(text): # roberta
  '''
  Classifies and returns the underlying emotion of a text string
  '''
  text = re.sub(r'[^\w\s]', '', text)
  text = text.lower()
  t = ByteLevelBPETokenizer(
            "tokenizer/vocab.json", #change path
            "tokenizer/merges.txt"  #change path
        )
  t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
  t.enable_truncation(512)
  t.enable_padding(pad_id=t.token_to_id("<pad>"))
  tokenizer = t
  encoded = tokenizer.encode(text)
  sequence_padded = torch.tensor(encoded.ids).unsqueeze(0)
  attention_mask_padded = torch.tensor(encoded.attention_mask).unsqueeze(0)
  with torch.no_grad():
      output = emo_model((sequence_padded, attention_mask_padded))
  top_p, top_class = output.topk(1, dim=1)
  label = int(top_class[0][0])
  label_map = {v: k for k, v in label2int.items()}
  return label_map[label]



# def get_emotion(text):  # BERT
#     '''
#     Classifies and returns the underlying emotion of a text string
#     '''
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.lower()
#     encoded = bert_tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         return_token_type_ids=False,
#         padding="max_length",
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#     )
#     sequence_padded = torch.tensor(encoded["input_ids"])
#     attention_mask_padded = torch.tensor(encoded["attention_mask"])
#     with torch.no_grad():
#         output = emo_model((sequence_padded, attention_mask_padded))
#     top_p, top_class = output.topk(1, dim=1)
#     label = int(top_class[0][0])
#     # print(label)
#     label_map = {v: k for k, v in label2int.items()}
#     return label_map[label]


# def empathy_score(text):  # BERT
#     '''
#     Computes a discrete numerical empathy score for an utterance (scale 0 to 1)
#     '''

#     # text = re.sub(r'[^\w\s]', '', text)
#     # text = text.lower()
#     encoded = bert_tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         return_token_type_ids=False,
#         padding="max_length",
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#     )
#     sequence_padded = torch.tensor(encoded["input_ids"])
#     # print(sequence_padded)
#     attention_mask_padded = torch.tensor(
#         encoded["attention_mask"])
#     # print(attention_mask_padded)
#     with torch.no_grad():
#         output = emp_model((sequence_padded, attention_mask_padded))
#     top_p, top_class = output.topk(1, dim=1)
#     label = int(top_class[0][0])
#     return label

# def empathy_score(text): (T5)
#     '''
#     Computes a discrete numerical empathy score for an utterance (scale 0 to 2)
#     '''
#     with torch.no_grad():
#         input_ids = emp_model.tokenizer.encode(
#             text + '</s>', return_tensors='pt')
#         output = emp_model.model.generate(input_ids=input_ids, max_length=2)
#         dec = [emp_model.tokenizer.decode(ids) for ids in output]
#     label = dec[0]
#     if label == 'weak':
#         score = 0.0
#     elif label == 'strong':
#         score = 1.0
#     # normalise between 0 and 1 dividing by the highest possible value:
#     return score

def empathy_score(text):  # RoBERTa
    '''
    Computes a discrete numerical empathy score for an utterance (scale 0 to 1)
    '''

    # text = re.sub(r'[^\w\s]', '', text)
    # text = text.lower()
    t = ByteLevelBPETokenizer(
                "NLP models/Empathy Classification/tokenizer/vocab.json", #change path
                "NLP models/Empathy Classification/tokenizer/merges.txt"  #change path
            )
    t._tokenizer.post_processor = BertProcessing(
                ("</s>", t.token_to_id("</s>")),
                ("<s>", t.token_to_id("<s>")),
            )
    t.enable_truncation(512)
    t.enable_padding(pad_id=t.token_to_id("<pad>"))
    tokenizer = t
    encoded = tokenizer.encode(text)
    sequence_padded = torch.tensor(encoded.ids).unsqueeze(0)
    attention_mask_padded = torch.tensor(encoded.attention_mask).unsqueeze(0)
    with torch.no_grad():
        output = emp_model((sequence_padded, attention_mask_padded))
    top_p, top_class = output.topk(1, dim=1)
    label = int(top_class[0][0])
    return label

def perplexity(sentence):
    '''
    Computes the PPL of an utterance using GPT2 LM
    '''
    tokenize_input = gpttokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    with torch.no_grad():
        loss = gptmodel(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


def repetition_penalty(sentence):
    '''
    Adds a penalty for each repeated (stemmed) token in
    an utterance. Returns the total penalty of the sentence
    '''
    word_list = regextokenizer.tokenize(sentence.lower())
    filtered_words = [
        word for word in word_list if word not in stopwords.words('english')]
    stem_list = [stemmer.stem(word) for word in filtered_words]
    penalty = 0
    visited = []
    for w in stem_list:
        if w not in visited:
            visited.append(w)
        else:
            penalty += 0.001
    return penalty


def fluency_score(sentence):
    '''
    Computes the fluency score of an utterance, given by the
    inverse of the perplexity minus a penalty for repeated tokens
    '''
    ppl = perplexity(sentence)
    penalty = repetition_penalty(sentence)
    score = (1 / ppl) - penalty
    # normalise by the highest possible fluency computed on the corpus:
    normalised_score = score / 0.155
    if normalised_score < 0:
        normalised_score = 0
    return round(normalised_score, 2)


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
            (td.overlap.normalized_distance(s1grams, s2grams))*i)
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

def sentiment_score(sentence):
    """
    Compute the sentiment score of the sentence. Scale (-1, 1).
    and also normalized to scale of (0,1) by adding 1 and divide by 2. max(0.9836), min(-0.613)
    """
    analyzer = SentimentIntensityAnalyzer()
    sent_score = analyzer.polarity_scores(sentence)
    sent_score =  (sent_score["compound"] + 0.613)/ (0.9836+0.613)
    return sent_score

    
    
def get_sentence_score(sentence, dataframe):
    '''
    Calculates how fit a sentence is based on its weighted empathy, fluency
    and novelty values
    '''
    empathy = empathy_score(sentence)
    fluency = (math.log(fluency_score(sentence)) + 5)/5
    novelty = novelty_score(sentence, dataframe)
    sentiment = (math.log(1- sentiment_score(sentence)+0.00001) +6)/6 if (math.log(1- sentiment_score(sentence)+0.00001) +6)/6 >0 else 0
    score = 1.5*empathy + fluency + 1.5*novelty + 0.75*sentiment
    return score

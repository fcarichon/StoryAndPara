import pandas as pd
import os
import spacy 
from tqdm import tqdm
from collections import defaultdict 
#Import des tokenizers - sent_tokenize pour les phrases et word_tokenize pour les mots avec nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import numpy as np
from numpy.linalg import norm


import torch
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

class Conflict():
    
    def __init__(self, increase_ratio=2, n_window=1):
        
        self.sent_tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-sentiment")
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.sent_model = AutoModelForSequenceClassification.from_pretrained(f"cardiffnlp/twitter-roberta-base-sentiment")
        self.event_tokenizer = AutoTokenizer.from_pretrained("bhuvi/event_classification_model")
        self.event_model = AutoModelForSequenceClassification.from_pretrained("bhuvi/event_classification_model")
        self.nlp = spacy.load("en_core_web_sm")
        self.increase_ratio=increase_ratio
        self.n_window=n_window
    
    def truncate(self, text):
        
        token_list = [tok.text for tok in self.nlp.tokenizer(str(text))]
        if len(token_list) >= 350:
            token_list = token_list[0:350]
            
        return ' '.join(token_list)
    
    def conflict_event(self, text):
        """conflict -- no event detected then 0 | if conflict detected with neg sent == 1 | 2 if event but no negative sent
        Redo function per sentence
        
        """
        text = self.truncate(text)
        event_encoding = self.event_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=350)
        event_out = self.event_model(**event_encoding)
        event_scores = event_out[0][0].detach().numpy()
        event_scores = softmax(event_scores)
        #0 if negative for event | 1 if positiive
        event =  event_scores.argmax()
        
        conflict = 0
        if event == 1:
            sent_encoding = self.sent_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=350)
            sent_output = self.sent_model(**sent_encoding)
            sent_scores = sent_output[0][0].detach().numpy()
            sent_scores = softmax(sent_scores)
            #0 if negative for event | 1 if neutral sent | 2 if positiive
            if sent_scores.argmax() == 0:
                conflict = 1
            else:
                conflict = 2
                
        return conflict
    
    def sent_characters(self, text):
        
        """ Binary variable if negative sentiment when 2 characters or more are involoved
            Simplistic version relying on POS tagger"""
        
        char_conflict = 0
        doc = self.nlp(text)
        count_char = 0
        for token in doc:
            if token.pos_ in ["PRON", "PROPN"]:
                count_char += 1
                
        if count_char >= 2:
            sent_encoding = self.sent_tokenizer(text, return_tensors='pt')
            sent_output = self.sent_model(**sent_encoding)
            sent_scores = sent_output[0][0].detach().numpy()
            sent_scores = softmax(sent_scores)
            #0 if negative for event | 1 if neutral sent | 2 if positiive
            if sent_scores.argmax() == 0:
                char_conflict = 1
                
        return char_conflict
    
    def do_increase(self, sent_list):
        
        get_increase = 0
        for i in range(len(sent_list)):
            current = sent_list[i]
            #Getting independ windows list for the current sentiment
            if i < self.n_window:
                following = sent_list[i+1:i+self.n_window+1]
                preceding = sent_list[0:i]
            else: 
                following = sent_list[i+1:i+self.n_window+1]
                preceding = sent_list[i-self.n_window:i]
            
            #Checking increase
            if len(following) != 0:
                mean_following = sum(following) /len(following)
                #We could easily change l.126 issue by imposing that current > 0.5 to be sure increase and main dimension is negative
                if current > self.increase_ratio * mean_following:
                    get_increase = 1
            if len(preceding) != 0:
                mean_preceding = sum(preceding) /len(preceding)
                if current > self.increase_ratio * mean_preceding:
                    get_increase = 1
        return get_increase
        
    def increase_sent(self, text):
        
        """ Binary variable : Look if there is an increase in negative sentiment intensity compared to previous sentences"""
        
        sent_increase = 0
        sent_list = sent_tokenize(text)
        
        neg_list = []
        #We don't need to estimate increase if we only have one sentence
        if len(sent_list) >= 2:
            for sent in sent_list:
                sent_encoding = self.sent_tokenizer(sent, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)
                if sent_scores.argmax() == 0:
                    neg_list.append(sent_scores[0].item())
            #Becuase of the if -- it does mean we check increase compare to the n_previous negative sentences
            sent_increase = self.do_increase(neg_list)
                    
        return sent_increase
    
    def change_sent_status(self, text):
        
        change_sent_prop = 0
        sent_list = sent_tokenize(text)
        #We don't need to estimate change of feelings if we only have one sentence
        num_sent = []
        if len(sent_list) >= 2:
            for sent in sent_list:
                sent_encoding = self.sent_tokenizer(sent, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)
                num_sent.append(sent_scores.argmax())
            
            #everytime the next sentence change sentiment we coun it
            for i in range(len(num_sent)-1):
                if num_sent[i] != num_sent[i+1]:
                    change_sent_prop+=1
                    
        return change_sent_prop/len(sent_list)
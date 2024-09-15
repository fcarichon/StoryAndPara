import spacy 
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

class Causality():
    #Model used for textuel implication -- reference:
    # https://huggingface.co/roberta-large-mnli?text=The%2Bdog%2Bwas%2Blost.%2BNobody%2Blost%2Bany%2Banimal
    #classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
    
    def tokenize(self, text):
        return [tok.text for tok in self.nlp.tokenizer(str(text))]
    
    def truncate(self, text):
        
        token_list = [tok.text for tok in self.nlp.tokenizer(str(text))]
        if len(token_list) >= 350:
            token_list = token_list[0:350]
            
        return ' '.join(token_list)
    
    def causal_chorence(self, text):
        
        """pipeline give us a probability from 0 to 1 that a text can be used as a classifier by another.
           Input : TExt
           Output : Average score for the list of sentence in the text to measure their textual implications"""
        
        #Getting the list of all sentences
        sentence_list = sent_tokenize(text)
        
        #Special case - only one sentence
        if len(sentence_list) <= 1:
            causal_coeff = 0.0
            
        else:
            causal_list = []
            for i in range(len(sentence_list)-1):
                #classifier(sequence_to_classify, candidate_labels)
                causal_list.extend(self.classifier(sentence_list[i], sentence_list[i+1])["scores"])
            
            #Implication coefficient for the story is the average score for all sentences
            causal_coeff = sum(causal_list)/len(causal_list)
            
        return causal_coeff
    
    def causal_length(self, text, causal_thresh=0.5):
        
        sentence_list = sent_tokenize(text)
        max_length = 0
        #Special case - only one sentence
        if len(sentence_list) <= 1:
            return max_length
        else:
            causal_length = 0
            for i in range(len(sentence_list)-1):
                if self.classifier(sentence_list[i], sentence_list[i+1])["scores"][0] > causal_thresh:
                    causal_length += 1
                    if causal_length > max_length:
                        max_length = causal_length
                #If we do not have at some point a sentence that is implied by its predecessor, the current length is reset to 0. 
                #The objective is to keep the longest
                else:
                    causal_length = 0

            return max_length
        
    def causal_subord(self, text):
        
        """ 
        Count the number of coordination or surbodination terms per sentences.
        """
        
        sub_coord_list = ["But", "however", "so", "then", "thus", "therefore", "because", "accordingly", "consequently", "hence",
        "as a result of", "as long as", "as things go", "being", "by cause of", "by reason of", "by virtue of", "considering", "due to"
        "nonetheless", "notwithstanding", "yet", "after all", "all the same", "anyhow", "be that as it may", "for all that"
        "after", "although", "as", "as if", "as long as", "before", "despite", "even if", "even though", "if", "in order that", 
        "rather than", "since", "so that", "that", "though", "unless", "until", "when", "where", "whereas", "whether", "while",
        "provided that", "in order to", "whenever", "nor",  "than"]
        
        sentence_list = sent_tokenize(text)
        count = 0
        for sentence in sentence_list: 
            sentence_tok = self.tokenize(sentence)
            for conjonction in sub_coord_list:
                if conjonction in sentence_tok:
                    count += 1
        
        return count/len(sentence_list)
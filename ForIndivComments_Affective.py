import spacy
import torch
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

from collections import Counter
from collections import defaultdict 


class affect_relation():
    
    def __init__(self): 
        
        self.nlp = spacy.load("en_core_web_sm")
        self.sent_tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-sentiment")
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(f"cardiffnlp/twitter-roberta-base-sentiment")
        
        
    def sentiment(self, comments_list, feel_threshold=0.8):
        """final_sent"""
        sent_felt = []
        sent_intense = []
        for comment in comments_list:
            try:
                sent_encoding = self.sent_tokenizer(comment, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)

                #0 if negative for event | 1 if neutral sent | 2 if positiive
                if sent_scores.argmax() !=1:
                    sent_felt.append(1)
                else:
                    sent_felt.append(0)
                arousal_val = sent_scores[sent_scores.argmax()]
                if arousal_val > feel_threshold:
                    sent_intense.append(1)
                else:
                    sent_intense.append(0)
            except:
                sent_felt.append(0)
                sent_intense.append(0)

        return sent_felt, sent_intense
    
    def empathy(self, post, comments_list, sim_intens=0.1):
    
        count_same = []
        count_intens = []
        try:
            post_encoding = self.sent_tokenizer(post, return_tensors='pt')
            post_output = self.sent_model(**post_encoding)
            post_scores = post_output[0][0].detach().numpy()
            post_scores = softmax(post_scores)
            post_feeling = post_scores.argmax()
            arousal_post = post_scores[post_feeling]
        except:
            count_same = [0] * len(comments_list)
            count_intens = [0] * len(comments_list)

        #if post_feeling != 1: -- QUESTION A POSER A ADRIEN -- SI LE POST EST NEUTRE??
        for comment in comments_list: 
            try:
                sent_encoding = self.sent_tokenizer(comment, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)
                comment_feel = sent_scores.argmax()

                if comment_feel == post_feeling:
                    count_same.append(1)
                    comment_arousal = sent_scores[comment_feel]
                    if arousal_post-sim_intens < comment_arousal < arousal_post+ sim_intens:
                        count_intens.append(1)
                    else:
                        count_intens.append(0)
                else:
                    count_same.append(0)
                    count_intens.append(0)
            except:
                count_same.append(0)
                count_intens.append(0)

        return count_same, count_intens
    
    
    def contagion(self, post, comments_list):
    
        #cat_conta = 0
        conta_list = []
        try:
            post_encoding = self.sent_tokenizer(post, return_tensors='pt')
            post_output = self.sent_model(**post_encoding)
            post_scores = post_output[0][0].detach().numpy()
            post_scores = softmax(post_scores)
            post_feeling = post_scores.argmax()
        except:
            post_feeling = 1
            conta_list = [0] * len(comments_list)
            
        #If post is not neutral
        if post_feeling != 1:
            for comment in comments_list:
                try:
                    sent_encoding = self.sent_tokenizer(comment, return_tensors='pt')
                    sent_output = self.sent_model(**sent_encoding)
                    sent_scores = sent_output[0][0].detach().numpy()
                    sent_scores = softmax(sent_scores)
                    comment_feel = sent_scores.argmax()

                    if comment_feel == 1:
                        conta_list.append(-1)
                    else:
                        conta_list.append(1)
                except:
                    conta_list.append(0)
        else:
            conta_list = [0] * len(comments_list)

        return conta_list
    
    def nb_letter_repeat(self, comments_list):
    
        #Nb de token avec au moins deux lettre répétées
        repeat_list = []
        for comment in comments_list: 
            comment_tok = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            nb_repeat = 0
            for token in comment_tok:
                repeat_len = 0
                first_letter = True
                for i, letter in enumerate(token):
                    if letter == token[i-1]:
                        if first_letter:
                            repeat_len+=2
                        else:
                            repeat_len+=1
                        first_letter = False
                    else:
                        first_letter = True
                if repeat_len > 2:
                    nb_repeat +=1
            repeat_list.append(nb_repeat)
            
        return repeat_list

    
    def prop_upper(self, comments_list):
        #Proportion de majuscule vs minuscule 
        upper_list = []
        for comment in comments_list:
            nb_upper = 0
            nb_letters = 0
            for i, letter in enumerate(comment):
                if letter.isupper():
                    nb_upper += 1
                if letter != " ":
                    nb_letters += 1
            upper_list.append(nb_upper/nb_letters)

        return upper_list
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
        sent_felt = 0
        sent_intense = 0
        for comment in comments_list:
            try:
                sent_encoding = self.sent_tokenizer(comment, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)

                #0 if negative for event | 1 if neutral sent | 2 if positiive
                if sent_scores.argmax() !=1:
                    sent_felt += 1
                arousal_val = sent_scores[sent_scores.argmax()]
                if arousal_val > feel_threshold:
                    sent_intense += 1
            except:
                pass

        #Number of comments with feelings and with intense feelings
        avg_feel = sent_felt / len(comments_list)
        avg_intense = sent_intense/len(comments_list)

        return avg_feel, avg_intense
    
    def empathy(self, post, comments_list, sim_intens=0.1):
    
        count_same = 0
        count_intens = 0
        try:
            post_encoding = self.sent_tokenizer(post, return_tensors='pt')
            post_output = self.sent_model(**post_encoding)
            post_scores = post_output[0][0].detach().numpy()
            post_scores = softmax(post_scores)
            post_feeling = post_scores.argmax()
            arousal_post = post_scores[post_feeling]
        except:
            pass

        #if post_feeling != 1: -- QUESTION A POSER A ADRIEN -- SI LE POST EST NEUTRE??
        for comment in comments_list: 
            try:
                sent_encoding = self.sent_tokenizer(comment, return_tensors='pt')
                sent_output = self.sent_model(**sent_encoding)
                sent_scores = sent_output[0][0].detach().numpy()
                sent_scores = softmax(sent_scores)
                comment_feel = sent_scores.argmax()

                if comment_feel == post_feeling:
                    count_same += 1
                    comment_arousal = sent_scores[comment_feel]
                    if arousal_post-sim_intens < comment_arousal < arousal_post+ sim_intens:
                        count_intens += 1
            except:
                pass
        
        avg_epmpathy = count_same/len(comments_list)
        avg_same_intens = count_intens/len(comments_list)

        return avg_epmpathy, avg_same_intens
    
    
    def contagion(self, post, comments_list):
    
        cat_conta = 0
        try:
            post_encoding = self.sent_tokenizer(post, return_tensors='pt')
            post_output = self.sent_model(**post_encoding)
            post_scores = post_output[0][0].detach().numpy()
            post_scores = softmax(post_scores)
            post_feeling = post_scores.argmax()
        except:
            post_feeling = 1

        #If post is not neutral
        if post_feeling != 1:
            conta_list = []
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
                    pass
            #Getting the most common sentiment from the contagion list
            cat_conta = max(set(conta_list), key=conta_list.count)


        return cat_conta
    
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

        repeat_avg = sum(repeat_list)/len(comments_list)

        return repeat_avg
    
    
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

        prop_upper = sum(upper_list)/len(comments_list)

        return prop_upper
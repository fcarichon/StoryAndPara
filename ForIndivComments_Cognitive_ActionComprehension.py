import spacy
import numpy as np
from numpy.linalg import norm

import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import AutoTokenizer

from scipy.special import softmax
import torch

class comprehension():
    
    def __init__(self, ban_list = ['DET', 'PUNCT', 'CCONJ', 'CONJ','ADP', 'ADV', 'NUM']): 
        
        self.nlp = spacy.load("en_core_web_sm")
        self.ban_list = ban_list
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        
    def cosine_sim(self, a, b):
        return np.dot(a, b)/(norm(a)*norm(b))
    
    def tokenize(self, text):
        return [tok.text for tok in self.nlp.tokenizer(str(text))]
    
    #Average similarity of post and all comments
    def semantic_sim(self, post_text, comments_list):

        #Creating a vector representation of the document :
        doc_post = self.nlp(str(post_text))
        init_post_vec = np.zeros(doc_post[0].tensor.shape)
        len_ = 0
        for token in doc_post:
            if not token.is_stop and token.pos_ not in self.ban_list:
                init_post_vec += token.tensor
                len_ += 1

        #Return the value for all comments here not only for posts
        if len_ ==0:
            sim_list = [0.] * len(comments_list)
            return sim_list

        doc_vec = init_post_vec /len_

        sim_list = []
        for comment in comments_list:
            doc_comment = self.nlp(str(comment))
            init_comment_vec = np.zeros(doc_comment[0].tensor.shape)
            len_c = 0
            for token in doc_comment:
                if not token.is_stop and token.pos_ not in self.ban_list:
                    init_comment_vec += token.tensor
                    len_c += 1

            # Gérer les cas ou les listes sont vides
            if len_c ==0:
                sim_list.append(0.)
            else:
                comment_vec = init_comment_vec /len_c
                sim_list.append(self.cosine_sim(doc_vec, comment_vec))

        return sim_list
    
    def repeated_kw(self, post_text, comments_list):
    
        post_tokens = self.tokenize(post_text)

        repeat_comments = []
        for comment in comments_list:
            count_rep = 0
            comment_token = self.tokenize(comment)
            for token in comment_token:
                if token in post_tokens:
                    count_rep += 1
                count_rep = count_rep / len(comment_token)
            repeat_comments.append(count_rep)

        return repeat_comments
    
    def comment_predicted(self, post_text, comments_list):
    
        nsp_list = []
        for comment in comments_list:
            try:
                tokenized = self.tokenizer(post_text, comment, return_tensors='pt')
                output = self.model(**tokenized)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                nsp = 1 - scores.argmax() ## If 0 -- then next predicted : We 1-scores to append 1 if next sentence is predicted to have a coherent average
                nsp_list.append(nsp)
            except:
                nsp_list.append(0)

        #avg_nsp = sum(nsp_list)/len(comments_list)
        return nsp_list
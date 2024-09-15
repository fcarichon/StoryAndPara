import pandas as pd
import spacy
import torch
from nltk.corpus import wordnet

import torch
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

class attention():
    
    def __init__(self, keep_list = ["VERB"]): 
        
        self.keep_list = keep_list
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline('zero-shot-classification', model='roberta-large-mnli', device=0)
        
    #tokenizer as the one employed by the model 
    def vb_tokenizer(self, text):
        token_text = []
        doc = self.nlp(str(text))
        for token in doc:
            if token.pos_ in self.keep_list:
                token_text.append(str(token.text))

        return token_text
    
    # Post_Text -- extraire les vb d'actions qui ne sont pas des auxiliares
    def attentive_vb(self, post_text, comments_list):

        post_vb = self.vb_tokenizer(post_text)
        syns_p = []
        if len(post_vb) > 0:
            for verb in post_vb:
                syn_ = wordnet.synsets(verb)
                if len(syn_) >0:
                    syns_p.append(syn_[0])  #0 - we keep the main meaning of the verb
        
        ######################## MOYENNE MAX et MOYENNE MEAN PAR UTILISATEUR
        avg_min_all_com = []
        avg_avg_all_com= []
        if len(syns_p) > 0:
            for comment in comments_list:
                comment_vb = self.vb_tokenizer(comment)
                if len(comment_vb) > 0:
                    sim_per_comment = []
                    for verb_c in comment_vb:
                        if len(wordnet.synsets(verb_c)) >0:
                            syn_c = wordnet.synsets(verb_c)[0]
                            for syn_p in syns_p:
                                sim_per_comment.append(syn_c.wup_similarity(syn_p))
                    
                    if len(sim_per_comment) !=0:
                        avg_min_all_com.append(min(sim_per_comment))
                        avg_avg_all_com.append(sum(sim_per_comment) / len(sim_per_comment))
                    else:
                        avg_min_all_com.append(0.)
                        avg_avg_all_com.append(0.)
                else:
                        avg_min_all_com.append(0.)
                        avg_avg_all_com.append(0.)

        else:
            avg_min_all_com = [0.] * len(comments_list)
            avg_avg_all_com = [0.] * len(comments_list)

        return avg_min_all_com, avg_avg_all_com
    
    def follow_previous(self, post_id, df_post, df_comments):
    
        #GEtting the username and the date linked to the post
        temp = df_post[df_post["PostSysID"] == post_id]
        user_name = temp["PostUser"].iloc[0]
        post_date = temp["PostDate"].iloc[0]

        #Getting the dataframe of all posts associated to the user
        df_user = df_post[df_post["PostUser"] == user_name]
        #Getting post_id list of previous posts of that user
        post_id_prev = df_user[df_user["PostDate"] < post_date]["PostSysID"]

        #GEtting the current user list -- average of reposts
        user_list = list(df_comments[df_comments["PostSysID"] == post_id]["CommentUser"])

        #Getting the list of users related to comments on previous posts
        df_comments_prev = df_comments[df_comments["PostSysID"].isin(post_id_prev)]
        prev_user_list = list(df_comments_prev["CommentUser"])

        #For each user - counting the number of times that user appears in the comments of previous posts
        avg_user_count = []
        for user in user_list:
            count_prev = prev_user_list.count(user)
            avg_user_count.append(count_prev)

        #We return the number for each comments -- the lenght of this list should be equal to the number of comments of a post
        return avg_user_count
    
    def post_implied(self, post_text, comments_list):
    
        implication_list = []
        for comment in comments_list:
            try:
                implication = self.classifier(post_text, comment)["scores"]
                implication_list.append(implication[0])
            except:
                print(post_text, comment)
                implication_list.append(0.)

        #avg_implication = sum(implication_list) / len(implication_list)
        return implication_list
import pandas as pd
import nltk
import spacy

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenizer
import string
import re

class evaluation():
    
    def __init__(self, subj_path = "data/subjectivity_clues.csv", INVALID_POS=["CC", "CD", "DT", "EX", "IN", "LS", "PDT", "POS", "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WRB"]): 
        
        self.subj_references = pd.read_csv(subj_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.INVALID_POS = INVALID_POS
        
    def most_frequent(self, lst):
        return max(set(lst), key=lst.count)
    
    def tokenize(self, sentence):

        sentence = re.sub(f"[{re.escape(string.punctuation)}\…]+", " ", sentence)
        # Filter common words
        tokens = nltk.pos_tag(sentence.split())
        tokens_ = tokens
        tokens = [tok for (tok, pos) in tokens if tok.lower() not in self.stop_words and pos not in self.INVALID_POS]
        return tokens

    def subj_eval(self, comments_list):
    
        n_weak_scores, n_strong_scores = [], []

        for i, comment in enumerate(comments_list):
            n_weak_score, weak_score = 0., 0.
            n_strong_score, strong_score = 0., 0.
            token_list = [tok.text for tok in self.nlp.tokenizer(str(comment))]
            
            #print(self.subj_references.columns)
            for token in token_list:
                df_temp = self.subj_references[self.subj_references["words"]==token]
                if len(df_temp) > 0:
                    type_ = self.most_frequent(list(df_temp["Type"]))
                    if type_ == "weaksubj":
                        n_weak_score += 1/len(token_list)
                    if type_ == "strongsubj":
                        n_strong_score += 1/len(token_list)

            n_weak_scores.append(n_weak_score)
            n_strong_scores.append(n_strong_score)

        avg_weak = sum(n_weak_scores) / len(comments_list)
        avg_strong = sum(n_strong_scores) / len(comments_list)

        return avg_weak, avg_strong
    
    def senti_lookup(self, word):
        synsets = list(swn.senti_synsets(word))
        avg_pos, avg_neg, avg_obj = 0.0, 0.0, 0.0
        size = len(synsets)
        if size == 0:
            raise Error(f"'{word}' not found.")

        for synset in synsets:
            pos, neg, obj = synset.pos_score(), synset.neg_score(), synset.obj_score()
            avg_pos += pos/size
            avg_neg += neg/size
            avg_obj += obj/size

        return round(avg_pos, 5), round(avg_obj, 5), round(avg_neg, 5)
    
    def eval_neutrality(self, pos, obj, neg):
            THRESHOLD = 0.5
            if obj >= THRESHOLD:
                return 1
            return 0
        
    def eval_objectivity(self, pos, obj, neg):
        GAP_THRESHOLD = 0.65
        OBJ_THRESHOLD = 0.65
        gap = abs(pos - neg) / max(pos, neg)
        if obj <= OBJ_THRESHOLD and gap >= GAP_THRESHOLD:
            return 1
        return 0
    
    def calc_metrics_1(self, text):
        
        tokens = self.tokenize(text)
        #print(tokens)
        size = len(tokens) + 1
        n_neutral = 0
        n_biased = 0
        for token in tokens:
            try:
                pos, obj, neg = self.senti_lookup(token)
                #print(self.senti_lookup(token))
                n_neutral += self.eval_neutrality(pos, obj, neg)
                n_biased += self.eval_objectivity(pos, obj, neg)
            except:
                pass

        neu_score = n_neutral / size
        obj_score = 1.0 - (n_biased / size)

        return round(neu_score, 5), round(obj_score, 5)
    
    def calc_metrics_2(self, text):
        neu_score = 0.0
        obj_score = 0.0
        sentences = re.split(f"[{re.escape(string.punctuation)}\…]+", text)
        for sent in sentences:
            tokens = self.tokenize(text)
            size = max(len(tokens), 1)
            avg_pos, avg_obj, avg_neg = 0.0, 0.0, 0.0

            for token in tokens:
                try:
                    pos, obj, neg = self.senti_lookup(token)
                    avg_pos += pos
                    avg_obj += obj
                    avg_neg += neg
                except:
                    pass

            avg_pos = avg_pos / size
            avg_obj = avg_obj / size
            avg_neg = avg_neg / size

            # Neutrality test
            if avg_obj >= avg_pos and avg_obj >= avg_neg:
                neu_score += 1.0

            # Objectivity test
            THRESHOLD = 0.1
            if abs(avg_pos - avg_neg) <= THRESHOLD:
                obj_score += 1.0

        neu_score = neu_score / len(sentences)
        obj_score = obj_score / len(sentences)

        return round(neu_score, 5), round(obj_score, 5)
    
    def calc_scores(self, corpus, method):
        '''
        corpus: list of summaries
        method: m-1 or m-2
        '''
        avg_neu_score = 0
        avg_obj_score = 0
        #corpus_size = corpus.shape[0]
        corpus_size = len(corpus)
        for text in corpus:
            if method == "m-1":
                neu_score, obj_score = self.calc_metrics_1(text)
            elif method == "m-2":
                neu_score, obj_score = self.calc_metrics_2(text)
            else:
                raise Error(f"{method} is not supported. Try either 'm-1' or 'm-2'")

            avg_neu_score += neu_score / corpus_size
            avg_obj_score += obj_score / corpus_size

        return round(avg_neu_score, 5), round(avg_obj_score, 5)
    
    
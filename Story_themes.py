import spacy 
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import numpy as np
from numpy.linalg import norm

import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# Filtering all pos execpt NN and VB from texts
class Themes():
    
    def __init__(self, text_list, num_k=30, ban_list = ['DET', 'PUNCT', 'AUX'], alpha=0.1, beta=0.1):
        self.nlp = spacy.load("en_core_web_sm")
        self.ban_list = ban_list
        self.num_k = num_k
        self.train_list = text_list
        self.train_lda(self.train_list, self.num_k)
        
    #tokenizer as the one employed by the model 
    def filtered_tokenize(self, text):
        token_text = []
        doc = self.nlp(str(text))
        for token in doc:
            if not token.is_stop and token.pos_ not in self.ban_list:
                token_text.append(str(token.text))

        return token_text

    def train_lda(self, text_list, num_k):

        sent_token = []
        for text in text_list:
            sentence_list = sent_tokenize(text)
            for sent in sentence_list:
                sent_token.append(self.filtered_tokenize(sent))
            
        self.common_dictionary = Dictionary(sent_token)
        common_corpus = [self.common_dictionary.doc2bow(text) for text in sent_token]
        self.lda = LdaModel(common_corpus, id2word=self.common_dictionary, num_topics=num_k, )
    
    def topic_vectorization(self, gensim_corpus):
        
        """ Creating topic vectors once we have train the lda model
            TO ADD : LOAD A PRETRAINED MODEL ONSTEAD OF LAUNCHING THE TRAIN_LDA EVERYTIME"""
        
        full_vector_sums = []
        for unseen_text in gensim_corpus:
            vector_ = self.lda[unseen_text]
            topic_idx, topic_value = zip(*vector_)
            vector = []
            for i in range(self.num_k):
                if i in topic_idx:
                    idx = topic_idx.index(i)
                    vector.append(topic_value[idx])
                else:
                    vector.append(0.)
            full_vector_sums.append(vector)

        return full_vector_sums
        
    def diversity(self, text):
        """DEux cas ici : 
            Le score de diversité doit être de 0 si on a qu'une seule phrases - score inter-sententiel.
            Sinon, on cluster avec DBSCN les phrases? et on regarde la taille du cluster principal sur le nb de cluster au total
            Input MUST BE DIFFRENT FROM INPUT LIST IN CLASS FOR TRAIN/TEST DATASETS
            Output : Ratio of number of different main topics per total number of sentences -- the closer to 1 the higher the diversity, the closer to 0 the less"""
        
        sent_list = sent_tokenize(text)
        diversity = 0
        if len(sent_list) >= 2:
            token_list = []
            for text in sent_list:
                token_list.append(self.filtered_tokenize(text))

            sent_corpus = [self.common_dictionary.doc2bow(text) for text in token_list]
            vector_sent_corpus = self.topic_vectorization(sent_corpus)
            #Calating the diversity of topics here
            best_topic_list = []
            for i in range(len(vector_sent_corpus)):
                best_topic_list.append(np.argmax(vector_sent_corpus[i]))
            #Compute number of unique value of topics -- number of different topics
            nb_diff_main_topics = len(set(best_topic_list))
            diversity = nb_diff_main_topics / len(sent_list)
        
        return diversity
    
    def cosine_sim(self, a, b):
        return np.dot(a, b)/(norm(a)*norm(b))
    
    def homogeneity(self, text):
        
        """Computing intra-sentence homogeneity of text"""
        
        #defining the uniform distribution vector
        uniform_topic_dist = np.array([1/self.num_k] * self.num_k)
        sent_list = sent_tokenize(text)
        token_list = []
        for text in sent_list:
            token_list.append(self.filtered_tokenize(text))
            
        sent_corpus = [self.common_dictionary.doc2bow(text) for text in token_list]
        vector_sent_corpus = self.topic_vectorization(sent_corpus)
        hom_dist = []
        for i in range(len(vector_sent_corpus)):
            hom_dist.append(self.cosine_sim(vector_sent_corpus[i], uniform_topic_dist))
            
        homogeneity = 1- (sum(hom_dist) / len(hom_dist))

        return homogeneity
    
    # Best matching topic & vocabulary proximity
    def term_coverage(self, sent1, sent2):
        count = 0.
        for word_idx, word in enumerate(sent1):
            if word in sent2:
                sent2_idx = sent2.index(word)
                score = 1/(word_idx+1) * 1/(sent2_idx+1)
                count += score
        return count
    
    def overlap(self, sent_1, sent_2, top=50):
        
        best_topic_sent1 = np.argmax(sent_1)
        best_topic_sent2 = np.argmax(sent_2)
        
        #Getting the term of the topic associated with the first sentence
        ref_terms = self.lda.get_topic_terms(best_topic_sent1, topn=top)
        word_ref_idx, word_ref_value = zip(*ref_terms)
        
        #Computing overlap score
        over_terms = self.lda.get_topic_terms(best_topic_sent2, topn=top)
        word_over_idx, word_over_value = zip(*over_terms)
        score_cover = self.term_coverage(word_ref_idx, word_over_idx)
        
        #return ratio of overlapping terms on the total number of term describing the topic
        return score_cover / top
        
    def consistence(self, text, top_cons=20):
        
        """On prend le score principal et regarde le word overlap des termes - score inter-sententiel """
        sent_list = sent_tokenize(text)
        consistency = 0
        if len(sent_list) >= 2:
            token_list = []
            for text in sent_list:
                token_list.append(self.filtered_tokenize(text))

            sent_corpus = [self.common_dictionary.doc2bow(text) for text in token_list]
            vector_sent_corpus = self.topic_vectorization(sent_corpus)
            
            sent_consist = []
            for i in range(len(vector_sent_corpus)-1):
                sent_consist.append(self.overlap(vector_sent_corpus[i], vector_sent_corpus[i+1], top=top_cons))
            
            consistency = sum(sent_consist) / len(sent_consist)
            
        return consistency
import spacy 
from collections import defaultdict 
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class Characters():
    
    def __init__(self):
        
        self.nlp = spacy.load("en_core_web_sm")
        
        self.first_sing = ['i', 'my', 'me', 'myself', 'mine']
        self.first_plur = ['we', 'our','us', 'ourselves', 'ours']
        self.second = ['you', 'your', 'u', 'ya', 'yourself', 'yours', 'ur', "yer'", 'urself', 
                          "ya'll", 'yourselves', 'thy', 'yaself', "y'all"]
        self.third_sing = ['it', 'he', 'his','her','she', 'its', 'someone', 'him', 'somebody', 'himself', 'herself','itself', 'hers']
        self.third_plur = ['they', 'their', 'everyone', 'them','em', 'everybody','themselves', "'em", "theirs"]
        self.valid_ent = ["PERSON"]
        
    def pronouns(self, text):
        
        """ Create a dictionnary of binary variable all indacting if a type of pronoun is present in the text
            Input: text
            Output: dictionnary {binary value - 1 for presence | 0 for absence} """
        
        binary_pron_dict = {}
        doc = self.nlp(text)
        binary_pron_dict = {"first_sing": 0, "first_plur":0, "second":0, "third_sing":0, "third_plur":0} 
        for token in doc:
            token_text = token.text.lower()
            if token.pos_ == "PRON" and  token_text in self.first_sing:
                binary_pron_dict["first_sing"] = 1
            if token.pos_ == "PRON" and  token_text in self.first_plur:
                binary_pron_dict["first_plur"] = 1
            if token.pos_ == "PRON" and  token_text in self.second:
                binary_pron_dict["second"] = 1
            if token.pos_ == "PRON" and  token_text in self.third_sing:
                binary_pron_dict["third_sing"] = 1  
            if token.pos_ == "PRON" and  token_text in self.third_plur:
                binary_pron_dict["third_plur"] = 1
        
        return binary_pron_dict
        
    def named_entities(self, text):
        """ Create a binary variable indicating the presence or absence of Named entity in a text that is a person or an organisation.
            Input: text
            Output binary value - 1 for presence | 0 for absence """
        
        doc_ent = self.nlp(text)
        for ent in doc_ent.ents:
            if ent.label_ in self.valid_ent:
                return 1
            else:
                pass
        return 0
    
    def most_used_pron(self, text):
        
        """ Count number of pronouns in sentence and return a categorical variable associated with the given category
        Input : text
        Ouput : 0 == No pronouns | 1 == first person sing pronouns | 2 == first person plur. pron. | 3 == second pers. | 4 == third pers. sing. | 5 == thirs pers. plur.
        """
        
        pronoun_dict = defaultdict(int)
        doc = self.nlp(text)
        for token in doc:
            token_text = token.text.lower()
            if token.pos_ == "PRON" and token_text in self.first_sing:
                pronoun_dict["first_sing"] += 1
            if token.pos_ == "PRON" and  token_text in self.first_plur:
                pronoun_dict["first_plur"] += 1
            if token.pos_ == "PRON" and token_text in self.second:
                pronoun_dict["second"] += 1
            if token.pos_ == "PRON" and token_text in self.third_sing:
                pronoun_dict["third_sing"] += 1
            if token.pos_ == "PRON" and token_text in self.third_plur:
                pronoun_dict["third_plur"] += 1
                
        if len(pronoun_dict) == 0:
            return 0
        
        ### If same value the first person is selected and so on...
        else:
            key_ = max(pronoun_dict, key=pronoun_dict.get)
            if key_ == "first_sing":
                return 1
            if key_ == "first_plur":
                return 2
            if key_ == "second":
                return 3
            if key_ == "third_sing":
                return 4
            if key_ == "third_plur":
                return 5
            
    def prop_same_sent(self, text):
        
        """ Define the proportions of sentences including at least two types of different pronouns
        Input : text
        Output : Ratio - Nb of sentence with 2 different pronouns / total nb of sentence
        """
        
        count = 0
        sentences = sent_tokenize(text)
        #For each sentence we initialize a count for pronouns, if we reach 2 different categories then we have two characters, then we count for the sentences
        for sentence in sentences:
            count_pronEN = []
            doc_sent = self.nlp(sentence)
            for token in doc_sent:
                token_text = token.text.lower()
                if token_text in self.first_sing:
                    count_pronEN.append("first_s")
                if token_text in self.first_plur:
                    count_pronEN.append("first_p")
                if token_text in self.second:
                    count_pronEN.append("second")
                if token_text in self.third_sing:
                    count_pronEN.append("thrid_s")
                if token_text in self.third_plur:
                    count_pronEN.append("thrid_p")
            for ent in doc_sent.ents:
                if ent.label_ == "PERSON":
                    count_pronEN.append(ent.text)
            
            ## Accounting for multiple pronouns
            if len(set(count_pronEN)) >=2 :
                count+=1
            
        return count / len(sentences)
    
    
    def EN_as_subj(self, text):
        
        doc_text = self.nlp(text)
        accepted_pronouns = self.first_sing + self.first_plur + self.second + self.third_sing + self.third_plur  
        for token in doc_text:
            if token.dep_ == "ROOT":
                root_token = token.text
                
        for token in doc_text:
            if token.dep_ == "nsubj":
                if token.head == root_token:
                    if token.pos_ in accepted_pronouns:
                        return 2
                    elif token.text_ in str(doc_text.ents):
                        return 1
                    else:
                        return 0

        return 0
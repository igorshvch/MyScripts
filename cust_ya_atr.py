import math

class Ranger():
    def __init__(self, math_log = math.log10):
        self.D = None
        self.paths = None
        self.math_log = math_log
    
    def extract_common_freq(self, word):
        ...
    
    def extract_term_freq(self, word, doc):
        ...
    
    def extract_p_ontopic(self, word):
        CF = self.extract_common_freq(word)
        return 1-math.exp(-1.5*CF/self.D)

    def w_single(self, word, doc):
        DL = len(doc)
        TF = self.extract_term_freq(word, doc)
        p_ontopic = self.extract_p_ontopic(word)
        return self.math_log(p_ontopic)*(TF/TF+1+(1/350)+DL)
    
    def extract_pairs_term_freq(self, word1, word2, doc):
        ...
    
    def w_pair(self, word1, word2, doc):
        p_ontopic1 = self.extract_p_ontopic(word1)
        p_ontopic2 = self.extract_p_ontopic(word2)
        TF = self.extract_pairs_term_freq(word1, word2, doc)
        return (
            0.3*(self.math_log(p_ontopic1)+self.math_log(p_ontopic2))*TF/1+TF
        )
    
    def all_words(self, phrase, doc):
        if isinstance(phrase, str):
            phrase = phrase.split()
        p_ontopics = [self.extract_p_ontopic(w) for w in phrase]
        ...


    

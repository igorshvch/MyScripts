#see sorce: https://github.com/chibueze07/Machine-Learning-In-Law/blob/master/project.ipynb

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import mglearn

class SimpleTopicModel():
    def __init__(self, cleaned_text=None, topic_qn=5, words_qn=10, ngr=(1,1)):
        self.text = cleaned_text
        self.tq = topic_qn
        self.wq = words_qn
        self.ngr=ngr
        self.model()
    
    def model(self):
        vect=CountVectorizer(ngram_range=self.ngr)
        dtm=vect.fit_transform(self.text)
        dtm
        lda=LatentDirichletAllocation(n_components=self.tq)
        _=lda.fit_transform(dtm)
        sorting=np.argsort(lda.components_)[:,::-1]
        features=np.array(vect.get_feature_names())
        mglearn.tools.print_topics(topics=range(self.tq), feature_names=features, \
                                   sorting=sorting, topics_per_chunk=self.tq, n_words=self.wq)

import numpy as np
from time import time
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

C_VEC = CountVectorizer(token_pattern='\w+')
T_VEC = TfidfVectorizer(token_pattern='\w+')
    
def create_vectors(pars_lst, vectorizer):
    compressed_mtrx = vectorizer.transform(pars_lst)
    result_mtrx = compressed_mtrx.toarray()
    assert len(pars_lst) == result_mtrx.shape[0]
    return result_mtrx

def act_and_concl_to_mtrx(vector_pop='concl',
                          vector_model=None,
                          addition=True,
                          fill_val=1):
    '''
    Accepted 'vector_pop' args:
    'act', 'concl', 'mixed'
    Accepted 'vect_model' args:
    'count', 'tfidf'
    '''
    if vector_model == 'count':
        vectorizer = C_VEC
    elif vector_model == 'tfidf':
        vectorizer = T_VEC
    def inner_func1(pars_list, concl):
        data = [concl] + pars_list
        vectorizer.fit(data)
        data_mtrx = create_vectors(data, vectorizer)
        if addition:
            update_mtrx = (
                np.append(
                    data_mtrx,
                    np.full(
                        (len(data_mtrx),1), fill_val
                        ),
                    1
                )
            )
            return update_mtrx
        else:
            return data_mtrx
    def inner_func2(pars_list, concl, bigrs=None):
        data = [concl] + pars_list
        vectorizer.fit([concl])
        data_mtrx = create_vectors(data, vectorizer)
        if addition:
            update_mtrx = (
                np.append(
                    data_mtrx,
                    np.full(
                        (len(data_mtrx),1), fill_val
                        ),
                    1
                )
            )
            return update_mtrx
        else:
            return data_mtrx 
    def inner_func3(pars_list, concl, bigrs):
        data = [concl] + pars_list
        vectorizer.fit(data+[bigrs])
        data_mtrx = create_vectors(data, vectorizer)
        if addition:
            update_mtrx = (
                np.append(
                    data_mtrx,
                    np.full(
                        (len(data_mtrx),1), fill_val
                        ),
                    1
                )
            )
            return update_mtrx
        else:
            return data_mtrx   
    options = {
        'act' : inner_func1,
        'concl': inner_func2,
        'mixed': inner_func3
    }
    return options[vector_pop]

def eval_cos_dist(index_mtrx, output='best'):
    base = index_mtrx[0,:]
    holder = []
    for i in range(1,index_mtrx.shape[0],1):
        cos = sp_cosine(base, index_mtrx[i,:])
        cos = cos if not np.isnan(cos) else 1.0
        holder.append((i, cos))
    if output=='best':
        return sorted(holder, key = lambda x: x[1])[0]
    elif output=='all':
        return sorted(holder, key = lambda x: x[1])
    else:
        raise TypeError('Wrong key argument for "output"!')
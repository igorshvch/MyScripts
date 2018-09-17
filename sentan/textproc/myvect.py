import numpy as np
from time import time
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__version__ = '0.1'

###Content=====================================================================
C_VEC = CountVectorizer(token_pattern='\w+')
T_VEC = TfidfVectorizer(token_pattern='\w+')
    
def create_vectors(pars_lst, vectorizer):
    compressed_mtrx = vectorizer.transform(pars_lst)
    index_mtrx = compressed_mtrx.toarray()
    assert len(pars_lst) == index_mtrx.shape[0]
    return index_mtrx

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
    def inner_func1_act(pars_list, concl, pars_with_bigrs=None):
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
    def inner_func2_concl(pars_list, concl, pars_with_bigrs=None):
        concl = [concl]
        data = concl + pars_list
        vectorizer.fit(concl)
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
    def inner_func3_mixed(pars_list, concl, pars_with_bigrs):
        concl = [concl]
        vectorizer.fit(concl + pars_list)
        data_mtrx = create_vectors(concl + pars_with_bigrs, vectorizer)
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
        'act' : inner_func1_act,
        'concl': inner_func2_concl,
        'mixed': inner_func3_mixed
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


###Testing=====================================================================
if __name__ == '__main__':
    import sys
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print('Not implemented!')
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')
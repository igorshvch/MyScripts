
class A():
    def __init__(self):
        pass

    def concl_2gram(self, concl, stop_w=None, reps=False, result=None):
        concl_prep = self.CTP.full_process(
            concl,
            par_type=par_type,
            stop_w=stop_w
        )
        bigrms = self.CTP.create_2grams(concl_prep)
        if reps:
            bigrms = self.CTP.extract_repetitive_ngrams(bigrms)
        if result == 'intersection':
            concl_int_bigrs = self.CTP.intersect_2gr(concl, stop_w)
            return ' '.join(concl_prep+concl_int_bigrs)
        elif result =='join':
            return ' '.join(concl_prep+bigrms)
        elif result =='simple':
            return  ' '.join(bigrms)

    def concl_uniq_words(self, concl, stop_w=None):
        concl_prep = self.CTP.full_process(
            concl,
            par_type='parser1',
            stop_w=stop_w
        )
        return ' '.join(set(concl_prep))

class B():
    concl = None
    def __init__(self):
        options = {
            ('W_stpw', 'reps', 'bigrs_join', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=None, reps=True, result='join')
            ),
            ('WO_stpw', 'reps', 'bigrs_join', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=stop_w, reps=True, result='join')
            ),
            ('W_stpw', 'NOreps', 'bigrs_join', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=None, reps=False, result='join')    
            ),
            ('WO_stpw', 'NOreps', 'bigrs_join', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=stop_w, reps=False, result='join')    
            ),
            ('W_stpw', 'reps', 'bigrs_simple', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=None, reps=True, result='simple')
            ),
            ('WO_stpw', 'reps', 'bigrs_simple', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=stop_w, reps=True, result='simple')
            ),
            ('W_stpw', 'NOreps', 'bigrs_simple', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=None, reps=False, result='simple')    
            ),
            ('WO_stpw', 'NOreps', 'bigrs_simple', 'NOuniq'): (
                self.concl_2gram(concl, stop_w=stop_w, reps=False, result='simple')    
            ),
            #########
            ('W_stpw', 'NOreps', 'NObigrs', 'uniq'): (
                self.concl_uniq_words(concl, stop_w=None)
            ),
            ('WO_stpw', 'NOreps', 'NObigrs', 'uniq'): (
                self.concl_uniq_words(concl, stop_w=stop_w)
            )
        }




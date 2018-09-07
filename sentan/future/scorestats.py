from collections import Counter

#Collectiting stats information from list of scores (scorer.py)

__version__ = 0.1

###Content=====================================================================
def trunc(num, dgts=2):
    st = '{0:2.{1}f}'.format(num, dgts)
    return float(st)

def process(list_of_nums, dgts=3, threshold = 0.02):
    #Initialize local vars:
    ct = Counter()
    inner_list = list_of_nums
    #Initialize local funcs:
    local_trunc = trunc
    pr_data = [local_trunc(num, dgts=dgts) for num in inner_list]
    ct.update(pr_data)
    mean = ct.most_common(1).pop()
    while mean[0] < threshold:
        print(len(inner_list))
        ct = Counter()
        inner_list = inner_list[:int((len(inner_list)/2))]
        pr_data = [local_trunc(num, dgts=dgts) for num in inner_list]
        ct.update(pr_data)
        mean = ct.most_common(1).pop()
    return {
        'max' : max(pr_data),
        'min' : min(pr_data),
        'mean' : ct.most_common(1),
        'mean_part' : ct.most_common(1).pop()[1]/len(pr_data),
        'aver' : sum(pr_data)/len(pr_data),
        'deepness_lvl' : len(inner_list)
    }




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
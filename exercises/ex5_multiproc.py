from sentan.lowlevel import rwtool
from sentan.lowlevel.mypars import (
    tokenize as my_tok,
    lemmatize as my_lem
)

__version__ = 0.1

###Content=====================================================================
def processor(raw_text):
    lemmed = my_lem(my_tok(raw_text))
    rwtool.save_object(
        lemmed,
        'EXERCISE_lemmed',
        r'C:\Users\EA-ShevchenkoIS\TextProcessing'
    )
    print(r'C:\Users\EA-ShevchenkoIS\TextProcessing\EXERCISE_lemmed')


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
        elif sys.argv[1] == '-run':
            processor(sys.argv[2])
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')
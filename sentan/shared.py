__version__='0.0.1'

#CONTENT=======================================================================

def init_globs():
    global GLOBS
    GLOBS={}
    #GLOBS = {
    #    'root_struct': {
    #        'Root':None,
    #        'RawText':None,
    #        'Projects':None,
    #        'TEMP':None,
    #        'Common':None
    #    },
    #    'proj_struct': {
    #        'ActsBase':None,
    #        'TEMP':None,
    #        'Conclusions':None,
    #        'StatData':None,
    #        'Results':None
    #    },
    #    'proj_path':None,
    #    'proj_name':None,
    #    'old':None
    #}

def init_db():
    global DB
    DB = {
        'DivActs':None,
        'TLI':None
    }

init_globs()
init_db()


#TESTS=========================================================================
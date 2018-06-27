import pymorphy2 as pm2
import re as re
from imp import reload
print('Pymorpy2 was imported as pm2.\nReload function was imported from imp module.\nAll internals from re module were imported into startpack namespace.')
print('Pm2 version num:', pm2.__version__)

find_par_ref_1 = (
    '[Сс]т[А-я]*?\.* [0-9]+ [А-я]+ [A-zА-я] [0-9]+-[А-я][А-я]'
)

find_par_ref_2 = (
    'п\. [0-9]+ *-* *[0-9]* [А-я]+ [А-я]+'
)
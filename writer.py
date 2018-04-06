def writer(iterable_object, file_name_string, prefix='custome'):
    '''Accepts iterable objects containing strings
or a string object itselfe'''

    if prefix == 'custome':
        import datetime
        today = datetime.date.today

    
    if type(iterable_object) != str:
        with open('{}.txt'.format('MYWRITE_'+str(today())+'_'+file_name_string
                                  if prefix=='custome'
                                  else prefix+'_'+file_name_string),
                  mode='a') as file:
            for i in iterable_object:
                i = str(i) + '\n'
                file.write(i)
    else:
        with open('{}.txt'.format('MYWRITE_'+str(today())+'_'+file_name_string
                                  if prefix=='custome'
                                  else prefix+'_'+file_name_string),
                  mode='a') as file:
            file.write(iterable_object)
    print('OK')

def writer(s_object, file_name_string):
    '''Accepts iterable objects containing strings
or a string object itselfe'''
    
    if not isinstance(s_object, str):
        with open('{}.txt'.format(file_name_string), mode='a') as file:
            for i in s_object:
                i = str(i) + '\n'
                file.write(i)
    else:
        with open('{}.txt'.format(file_name_string), mode='a') as file:
            file.write(s_object)
    print('OK')

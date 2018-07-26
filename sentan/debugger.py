def index_string_to_dict(index_string_list):
    for item in index_string_list:
        key, val = item.split('#')
        dct = {key:set(val.split('='))}
        return dct

def index_string_to_dict_set(spam):
    for item in spam:
        key, val = item.split('#')
        val = [str(item) for item in val.split('=')]
        dct = {key:set(val)}
        return dct
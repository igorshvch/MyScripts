import pathlib as pthl

def create_acts_subdirs(path, interdir_name=''):
    p = pthl.Path(path)
    files = [fle for fle in p.iterdir()]
    subdirs = [p.joinpath(interdir_name, fle.stem) for fle in files]
    for subdir in subdirs:
        subdir.mkdir(parents=True, exist_ok=True)
    return subdirs

def create_new_file_paths(file_quant, path, suffix='.txt'):
    p = pthl.Path(path)
    file_paths = [
        p.joinpath(str(i)).with_suffix(suffix)
        for i in range(file_quant)
    ]
    return file_paths

def collect_exist_file_paths(top_dir, suffix='.txt'):
    holder = []
    def inner_func(top_dir, suffix):
        p = pthl.Path(top_dir)
        nonlocal holder
        store = [path_obj for path_obj in p.iterdir()]
        for path_obj in store:
            if path_obj.is_dir():
                inner_func(path_obj, suffix)
            elif path_obj.suffix == suffix:
                holder.append(path_obj)
    inner_func(top_dir, suffix)
    return holder



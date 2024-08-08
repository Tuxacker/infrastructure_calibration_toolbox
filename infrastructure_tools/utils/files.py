from pathlib import Path
from typing import Any, Optional
from json import load as json_load
from pickle import load as pickle_load, dump as pickle_dump
from yaml import safe_load as yaml_load


def load_file_impl_(path: Path):
    # print(f"Loading {path}")
    if path.suffix == ".yaml":
        with open(path, "r") as f:
            return yaml_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json_load(f)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle_load(f)
    else:
        raise NotImplementedError(f"{path.suffix} is not a supported file type!")


def load_file(path: Path, cache_suffix: Optional[str]=None):
    parent_directory = path.parent
    file_name = path.stem

    if cache_suffix is None:
        return load_file_impl_(path)
    else:
        if (parent_directory / (file_name + cache_suffix + ".pkl")).exists():
            return load_file_impl_(parent_directory / (file_name + cache_suffix + ".pkl"))
        else:
            return load_file_impl_(path)


def check_for_cache(path: Path, cache_suffix=""):
    return (path.parent / (path.stem + cache_suffix + ".pkl")).exists()


def create_cache(data, path: Path, cache_suffix=""):
    with open(path.parent / (path.stem + cache_suffix + ".pkl"), "wb") as f:
        pickle_dump(data, f)


def load_from_cache(path: Path, create_obj: Optional[Any]=None, cache_suffix: str="", verbose: bool=False):
    if check_for_cache(path, cache_suffix):
        data = load_file(path, cache_suffix)
        if verbose:
            print(f"Loaded {path.parent / (path.stem + '.pkl')} from cache")
    else:
        if create_obj is None:
            data = load_file(path)
        else:
            data = create_obj
        print(f"Loaded {path}, creating cache")
        create_cache(data, path, cache_suffix)
    return data
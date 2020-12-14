from os.path import exists, join
from os import mkdir, listdir
from typing import List
from pathlib import Path
import regex as re
import json


def maybe_create_dir(dir: str):
    if not exists(dir):
        mkdir(dir)


def list_full_paths(dir: str) -> List[str]:
    files = listdir(dir)
    paths = [join(dir, file) for file in files]
    return paths


def get_file_name(path: str) -> str:
    return Path(path).stem


def jsonc_load(path: str):
    text = open(path, "r", encoding="utf-8").read()
    return json.loads(re.sub("//.*", "", text, flags=re.MULTILINE))

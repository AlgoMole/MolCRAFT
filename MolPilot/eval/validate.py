import pickle
from pprint import pprint

import fire
import torch

def port_dict_format(data: dict) -> dict:
    template = {}
    for key, value in data.items():
        if isinstance(value, dict):
            template[key] = port_dict_format(value)
        else:
            template[key] = type(value)
    return template

def create_from_template(template:dict) -> dict:
    data = {}
    for key, val in template.items():
        if isinstance(val, dict):
            data[key] = create_from_template(val)
        else:
            data[key] = val()
    return data

def validate_dict_format(data: dict, template: dict, chain:str=""):
    for key, type_value in template.items():
        if isinstance(type_value, dict):
            if key not in data:
                print(f"{chain}[{key}] fail, last key not exists")
                return False
            elif not validate_dict_format(data[key], type_value, chain + f"[{key}]"):
                    return False
        else:
            if key not in data:
                print(f"{chain}[{key}] fail, last key not exists")
                return False
            elif not isinstance(data[key], type_value):
                print(f"{chain}[{key}] fail, type [{type(data[key])}] not match {type_value}")
                return False
    return True

def port_format(file:str):
    if file.endswith(".pt"):
        data = torch.load(file)
    elif file.endswith(".pkl"):
        with open(file, 'rb') as fd:
            data = pickle.load(fd)

    template = port_dict_format(data)
    pprint(template)

if __name__ == "__main__":
    fire.Fire(port_format)
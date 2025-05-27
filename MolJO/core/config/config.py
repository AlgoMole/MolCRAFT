import yaml
import os
import re
import fire
from ast import literal_eval
import argparse
import json
import copy


class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def __getitem__(self, index):
        return self.__dict__[index]


class Config:
    def __init__(self, config_file, **kwargs):
        _config = parse_config(path=config_file, subs_dict=kwargs)
        for key, value in _config.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, index):
        return self.__dict__[index]

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def save2yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.todict(), f, default_flow_style=False)

    def __str__(self):
        def prepare_dict4print(dict_):
            tmp_dict = copy.deepcopy(dict_)

            def recursive_change_list_to_string(d, summarize=16):
                for k, v in d.items():
                    if isinstance(v, dict):
                        recursive_change_list_to_string(v)
                    elif isinstance(v, list):
                        d[k] = (
                            (
                                str(
                                    v[: summarize // 2] + ["..."] + v[-summarize // 2 :]
                                )
                                + f" (len={len(v)})"
                            )
                            if len(v) > summarize
                            else str(v) + f" (len={len(v)})"
                        )
                    else:
                        pass

            recursive_change_list_to_string(tmp_dict)
            return tmp_dict

        return json.dumps(prepare_dict4print(self.todict()), indent=4, sort_keys=False)


def simplest_type(s):
    try:
        return literal_eval(s)
    except:
        return s


# enable parsing of environment variables in yaml files
def parse_config(
    path=None,
    data=None,
    subs_dict={},
    envtag="!ENV",
    substag="!SUB",
    envsubstag="!ENVSUB",
):
    # pattern for global vars: look for ${word}
    pattern = re.compile(".*?\${([\w:\-\.]+)}.*?")
    loader = yaml.FullLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(envtag, pattern, None)
    loader.add_implicit_resolver(substag, pattern, None)
    loader.add_implicit_resolver(envsubstag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                full_value = full_value.replace(
                    f"${{{g}}}", os.environ.get(vname, default_val)
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_subs_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        # print(value)
        # print(match)
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                if len(str(subs_dict.get(vname, default_val))) == 0:
                    raise ValueError(
                        f"""argument `{vname}` required, 
                        should be specified from command line with: --{vname} <value>,
                        or set a default value for `{vname}` in `{path}` file.
                        """
                    )
                full_value = full_value.replace(
                    f"${{{g}}}", str(subs_dict.get(vname, default_val))
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_envsubs_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        # print(value)
        # print(match)
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                if (
                    len(str(subs_dict.get(vname, default_val))) == 0
                    and len(str(os.environ.get(vname, default_val))) == 0
                ):
                    raise ValueError(
                        f"""argument `{vname}` required, 
                        should be specified from command line with: --{vname} <value>,
                        or set a default value for `{vname}` in `{path}` file.
                        """
                    )
                full_value = full_value.replace(
                    f"${{{g}}}",
                    str(subs_dict.get(vname, os.environ.get(vname, default_val))),
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_pathjoin(loader, node):
        value = loader.construct_sequence(node)
        value = [str(v) for v in value]
        return os.path.join(*value)

    def constructor_strjoin(loader, node):
        value = loader.construct_sequence(node)
        value = [str(v) for v in value]
        return "".join(value)

    def constructor_listadd(loader, node):
        value = loader.construct_sequence(node)
        value = sum([simplest_type(v) for v in value])
        return value

    def constructor_listmul(loader, node):
        value = loader.construct_sequence(node)
        ret = 1
        for v in value:
            ret *= simplest_type(v)
        return ret

    loader.add_constructor(envtag, constructor_env_variables)
    loader.add_constructor(substag, constructor_subs_variables)
    loader.add_constructor(envsubstag, constructor_envsubs_variables)
    loader.add_constructor("!PATHJOIN", constructor_pathjoin)
    loader.add_constructor("!STRJOIN", constructor_strjoin)
    loader.add_constructor("!LISTADD", constructor_listadd)
    loader.add_constructor("!LISTMUL", constructor_listmul)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError("Either a path or data should be defined as input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="../debug.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    _args, unknown = parser.parse_known_args()
    cfg = Config(**_args.__dict__)
    print(f"The config of this process is:\n{cfg}")

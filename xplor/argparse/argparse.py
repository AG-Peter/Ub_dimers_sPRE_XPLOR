################################################################################
# Imports
################################################################################

import os, sys

################################################################################
# Globals
################################################################################

__all__ = ['write_argparse_lines_from_yaml_or_dict']
YAML_FILE = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), "data/defaults.yml")

################################################################################
# Functions
################################################################################


def write_argparse_lines_from_yaml_or_dict(input='', print_argparse=False):
    """Uses the values from a yaml file to either build argparse lines, or build a string
    that sets these argparse lines.

    Keyword Args:
        yaml_file (Union[str, dict], optional): The path to the yaml file to parse.
            If empty string ('') is provided, the module's default yaml at xplor/data/defaults.yml will
            be used. Can also be fed with a dict. Defaults to ''.
        print_argparse (bool, optional): Whether to print argparse lines from
            that yaml. Defaults to False.

    Returns:
        str: A string, that can be provided to the XPLOR scirpt.

    """
    import yaml, subprocess

    flags = {}

    if not input:
        input = YAML_FILE
    if isinstance(input, str):
        with open(input, 'r') as stream:
            input = yaml.safe_load(stream)
    input_dict = input

    for pot in input_dict.keys():
        for arg_type in input_dict[pot].keys():
            for param, data in input_dict[pot][arg_type].items():
                flag = f'-{pot}_{arg_type}_{param}'
                required = 'False'
                type_ = f"{data['type']}"
                default = f"{data['value']}"
                help_ = f"{data['descr']}"
                if type_ == 'file':
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type=str, help="""{help_}""")'
                elif type_ == 'str':
                    flags[flag] = f'"{default}"'
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default="{default}", help="""{help_}""")'
                elif type_ == 'bool':
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type=str2bool, default="{default}", help="""{help_}""")'
                else:
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default={default}, help="""{help_}""")'
                if print_argparse: print(line)
    return ' '.join([f'{key} {val}' for key, val in flags.items()])
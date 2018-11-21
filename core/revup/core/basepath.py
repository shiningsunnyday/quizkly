import os
"""
Retrieve base path to load models later
"""
def get_base_path():
    """
    Returns module base path
    """
    return os.path.dirname(os.path.realpath(__file__))

"""This package contains modules related to function, network architectures, and models"""

import importlib
from .base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "model/[model_name]_model.py"."""
    model_file_name = "model." + model_name + "_model"
    modellib = importlib.import_module(model_file_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_file_name, model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model = find_model_using_name(model_name)
    return model.modify_options


def create_model(opt):
    """Create a model given the option."""
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

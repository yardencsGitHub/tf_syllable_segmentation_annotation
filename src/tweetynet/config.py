import ast

import attr


def two_element_tuple(instance, attribute, value):
    if type(value) != tuple:
        raise TypeError(
            f'{attribute} should be a two-element tuple but type was {type(value)}'
        )
    if len(value) != 2:
        raise TypeError(
            f'{attribute} should be a two-element tuple but had {len(value)} elements'
        )


def three_element_tuple(instance, attribute, value):
    if type(value) != tuple:
        raise TypeError(
            f'{attribute} should be a three-element tuple but type was {type(value)}'
        )
    if len(value) != 3:
        raise TypeError(
            f'{attribute} should be a three-element tuple but had {len(value)} elements'
        )


def str2tuple(value):
    if type(value) == tuple:
        return value
    elif type(value) == str:
        evald = ast.literal_eval(value)
        if type(evald) == tuple:
            return evald
        else:
            raise TypeError(
                f"conversion of string '{value}' to tuple instead produced type: {type(evald)}"
            )
    else:
        raise TypeError(
            f'invalid type for str2tuple converter: {type(value)}'
        )


@attr.s
class TweetyNetConfig:
    """config for TweetyNet model"""
    num_classes = attr.ib(converter=int)
    input_shape = attr.ib(converter=str2tuple, validator=three_element_tuple)
    conv1_filters = attr.ib(converter=int, default=32)
    conv1_kernel_size = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(5, 5))
    conv2_filters = attr.ib(converter=int, default=64)
    conv2_kernel_size = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(5, 5))
    conv_activation = attr.ib(converter=str, default='relu')
    l = attr.ib(converter=float, default=0.001)
    pool1_size = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(1, 8))
    pool1_strides = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(1, 8))
    pool2_size = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(1, 8))
    pool2_strides = attr.ib(converter=str2tuple, validator=two_element_tuple, default=(1, 8))
    lstm_dropout = attr.ib(converter=float, default=0.25)
    recurrent_dropout = attr.ib(converter=float, default=0.1)


def parse(num_classes, input_shape, model_config_section=None):
    config_kwargs = {
        'num_classes': num_classes,
        'input_shape': input_shape
    }
    if model_config_section is not None:
        config_kwargs.update(
            dict(model_config_section.items())
        )
    return TweetyNetConfig(**config_kwargs)

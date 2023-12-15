"""
`from iipyper.types import *` to get a set of types to use with `OSC.handle`
"""

from typing import NewType, List, Tuple, TypeVar, Iterable, Any, Dict, Optional, Union
from typing_extensions import TypeAliasType
import json

import numpy as np

import pydantic_numpy
from pydantic_numpy import NpNDArray as NDArray

class _Splat(type):
    # cache instances so that e.g. (Splat[None] is Splat[None]) evaluates True
    instances = {}
    @staticmethod
    def __getitem__(n):
        if n in _Splat.instances: return _Splat.instances[n]
        if n is None:
            Splat = NewType('Splat', List)
            r = Splat
        else:
            Splat = NewType('Splat', Tuple[(Any,)*n])
            r = Splat
        _Splat.instances[n] = r
        return r
    
class Splat(metaclass=_Splat):
    """horrible typing crimes to produce annotations for _consume_items
    which pydantic can also validate out of the box.

    `Splat[None]` aliases `List`

    `Splat[2]` aliases `Tuple[Any, Any]`

    `Splat[3]` aliases `Tuple[Any, Any, Any]`
    
    etc
    """

def ndarray_to_json(array: np.ndarray) -> str:
    array_list = array.ravel().tolist()
    array_info = {
        'data': array_list,
        'dtype': str(array.dtype),
        'shape': array.shape
    }
    return json.dumps(array_info)

def ndarray_from_json(json_str: str) -> np.ndarray:
    data = json.loads(json_str)
    array_data = data['data']
    dtype = data.get('dtype', 'float32')
    r = np.array(array_data, dtype=dtype)
    if 'shape' in data:
        r = r.reshape(data['shape'])
    return r

def ndarray_from_repr(repr_str: str) -> np.ndarray:
    from numpy import (array, 
        float16, float32, float64,
        complex64, complex128,
        int8, int16, int32, int64)
    return eval(repr_str)

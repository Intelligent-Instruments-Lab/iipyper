from typing import List, Tuple, TypeVar, Iterable, Any, Dict
from typing_extensions import TypeAliasType
import json

import numpy as np

import pydantic_numpy
from pydantic_numpy import NpNDArray as NDArray

class _Splat(type):
    instances = {}
    @staticmethod
    def __getitem__(n):
        if n in _Splat.instances: return _Splat.instances[n]
        if n is None:
            Splat = TypeAliasType('Splat', List)
            r = Splat
        else:
            N = TypeVar('N')
            Splat = TypeAliasType('Splat', Tuple[(Any,)*n], type_params=(N,))
            r = Splat[n]
        _Splat.instances[n] = r
        return r
class Splat(metaclass=_Splat):
    """horrible typing crimes to produce annotations for _consume_items
    which pydantic can also validate out of the box.

    Splat[None] aliases List
    Splat[2] aliases Tuple[Any, Any]
    Splat[3] aliases Tuple[Any, Any, Any]
    etc
    """

def ndarray_to_json(array: np.ndarray) -> str:
    array_list = array.tolist()
    array_info = {
        'data': array_list,
        'dtype': str(array.dtype),
        'shape': array.shape
    }
    return json.dumps(array_info)

def ndarray_from_json(json_str: str) -> np.ndarray:
    data = json.loads(json_str)
    array_data = data['data']
    dtype = data['dtype']
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

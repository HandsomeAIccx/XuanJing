# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 8:28 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : patch.py
# @Software: PyCharm
import copy

import numpy as np


class Patch(object):
    def __init__(self, **kwargs):
        super(Patch, self).__init__()
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        index_dict = Patch.index_dict(self.__dict__, index)
        index_patch = Patch()
        for k, v in index_dict.items():
            index_patch.__dict__[k] = v
        return index_patch

    def get_value(self, key):
        assert key in self.__dict__.keys(), f"{key} is invalid!"
        return self.__dict__[key]

    def __len__(self):
        if len(self.__dict__) == 0:
            return 0
        else:
            for k in self.__dict__.keys():
                return len(self.__dict__[k])

    def add(self, patch_data):
        assert patch_data.get_value('output')['act'].shape[
                   0] == 1, f"act shape == {patch_data.get_value('output')['act'].shape[0]}"
        assert isinstance(patch_data, Patch), "patch_data Must Be Patch Type."
        if self.__dict__ == {}:
            for k in patch_data.__dict__.keys():
                self.__dict__.update({k: patch_data.__dict__[k]})
        else:
            Patch.concatenate_dict(self.__dict__, patch_data.__dict__)

    def clear(self):
        self.__dict__.clear()

    @staticmethod
    def concatenate_dict(origin_dict, add_dict):
        add_dict_copy = copy.copy(add_dict)
        assert origin_dict.keys() == add_dict.keys(), f"{origin_dict}, {add_dict} dict keys error."
        for k in origin_dict.keys():
            if isinstance(add_dict[k], np.ndarray):
                origin_dict[k] = np.concatenate([
                    origin_dict[k], add_dict[k]
                ])
                assert add_dict_copy == add_dict, "add_dict != add_dict_copy!"
            elif isinstance(add_dict[k], dict):
                Patch.concatenate_dict(origin_dict[k], add_dict[k])
                assert add_dict_copy == add_dict, "add_dict != add_dict_copy!"
            else:
                raise TypeError(f"Not Support {type(add_dict[k])} type data")



    @staticmethod
    def index_dict(origin_dict, index):
        return_dict = {}
        for k, v in origin_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = v[index][np.newaxis, ]
            elif isinstance(v, dict):
                return_dict[k] = Patch.index_dict(v, index)
            else:
                raise TypeError(f"Not Support {type(origin_dict[k])} type data")
        return return_dict

if __name__ == "__main__":
    a = {"key1": {"key2": np.zeros((2, 3))}}
    b = {"key1": {"key2": np.zeros((2, 3))}}
    for i in range(10):
        Patch.concatenate_dict(a, b)
        print(a['key1']['key2'].shape)
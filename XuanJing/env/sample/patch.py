import copy
import numpy as np


class Patch(object):
    def __init__(self, **kwargs):
        super(Patch, self).__init__()
        self.__dict__.update(kwargs)

    def get_value(self, key):
        assert key in self.__dict__.keys(), f"{key} is invalid!"
        return self.__dict__[key]

    def add(self, patch_data):
        assert isinstance(patch_data, Patch), "patch_data Must Be Patch Type."
        if not self.__dict__:
            self.__dict__.update(patch_data.__dict__)
        else:
            Patch.concatenate_dict(self.__dict__, patch_data.__dict__)

    def clear(self):
        self.__dict__.clear()

    def __getitem__(self, index):
        index_dict = Patch.index_dict(self.__dict__, index)
        index_patch = Patch()
        for k, v in index_dict.items():
            setattr(index_patch, k, v)
        return index_patch

    def __len__(self):
        return 0 if not self.__dict__ else min([len(self.__dict__[k]) for k in self.__dict__.keys()])

    @staticmethod
    def concatenate_dict(origin_dict, add_dict):
        assert isinstance(origin_dict, dict) and isinstance(
            add_dict, dict
        ), "origin_dict and add_dict must be dictionaries."
        assert origin_dict.keys() == add_dict.keys(), f"{origin_dict}, {add_dict} dict keys error."
        for k, v in origin_dict.items():
            if isinstance(add_dict[k], np.ndarray):
                origin_dict[k] = np.concatenate([v, add_dict[k]])
            elif isinstance(add_dict[k], dict):
                Patch.concatenate_dict(v, add_dict[k])
            else:
                raise TypeError(f"Not Support {type(add_dict[k])} type data")

    @staticmethod
    def index_dict(origin_dict, index):
        assert isinstance(origin_dict, dict), "origin_dict must be a dictionary."
        return_dict = {}
        for k, v in origin_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = v[index][np.newaxis,]
            elif isinstance(v, dict):
                return_dict[k] = Patch.index_dict(v, index)
            else:
                raise TypeError(f"Not Support {type(origin_dict[k])} type data")
        return return_dict


if __name__ == "__main__":
    patch_instance_1 = Patch(
        obs=np.array((1, 2)), action=np.array((1, 1)), reward=np.array((1, 1)), next_obs=np.array((1, 2))
    )
    print(patch_instance_1.__dict__)
    patch_instance_2 = Patch(
        obs=np.array((1, 2)), action=np.array((1, 1)), reward=np.array((1, 1)), next_obs=np.array((1, 2))
    )
    patch_instance_1.add(patch_instance_2)
    print(patch_instance_2.__dict__)

from XuanJing.env.sample.patch import Patch
import numpy as np


def enhance_next_state(patch_list):
    assert isinstance(patch_list, list)
    post_data = Patch()
    for patch in patch_list:
        post_data.add(patch)
    episodes_reward = [sum(post_data.get_value("reward")) for post_data in patch_list]
    return post_data, episodes_reward
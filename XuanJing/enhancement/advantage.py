from XuanJing.env.sample.patch import Patch
import numpy as np

def enhance_advantage(patch_data):
    assert isinstance(patch_data, Patch)
    # obs = patch_data.__dict__['obs']
    # patch_data.__dict__['next_obs']
    return patch_data
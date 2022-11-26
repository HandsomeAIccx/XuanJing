def enhance_get_episode_reward(patch_data):
    assert isinstance(patch_data, list)
    # obs = patch_data.__dict__['obs']
    # patch_data.__dict__['next_obs']
    episodes_reward = [sum(patch.get_value('reward')) for patch in patch_data]
    return episodes_reward

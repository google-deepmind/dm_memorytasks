# Tasks

Each task has 3 different *levels* to run the agent on:

1.  **Train**
2.  **Holdout Interpolate**
3.  **Holdout Extrapolate**.

A further explanation of the per-task level split and details can be found in
our paper ([arxiv](https://arxiv.org/abs/1910.13406)).

## Passing a level name string to `dm_env` API

To run on a particular level, you need to append one of these suffixes to the
base task name.

1.  `_train`
2.  `_holdout_interpolate`
3.  `_holdout_extrapolate`

For example, you could train your agent on the `transitive_inference_train`
level and test on the `transitive_inference_holdout_interpolate` level.

We also provide 4 extra levels, not used in the paper, with the suffixes below:

*   `_interpolate`
*   `_extrapolate`

In these levels, the set of stimuli that is used is the same as in `_train`, but
the scale dimension is altered as per the corresponding holdout variant.

*   `_holdout_small`
*   `_holdout_large`

In these levels, the set of stimuli that is used is the holdout set, but the
scale dimensions are the ones used in `_train`.

To run on one of the 4 PsychLab tasks or the DeepMind Lab goal navigation tasks,
listed
[here](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/psychlab/memory_suite_01),
follow the
[DeepMind Lab instructions for using `dm_env`](https://github.com/deepmind/lab#train-an-agent).

### Base task names

#### PsychLab

*   `arbitrary_visuomotor_mapping`
*   `change_detection`
*   `continuous_recognition`
*   `what_then_where`

#### Spot the Difference (all in Unity)

*   `spot_diff`
*   `spot_diff_motion`
*   `spot_diff_multi`
*   `spot_diff_passive`

#### Goal Navigation (all in Unity except 1)

*   `explore_goal_locations` (DeepMind Lab)
*   `invisible_goal_empty_arena`
*   `invisible_goal_with_buildings`
*   `visible_goal_with_buildings`

#### Transitive Inference (in Unity)

*   `transitive_inference`

NOTE: For `explore_goal_locations` and `transitive_inference`, the **Train**
level was implemented as two separate files. Instead of appending `_train`,
append either `_train_small` or `_train_large`.

All task videos and descriptions can be found here:
[https://sites.google.com/corp/view/memory-tasks-suite](https://sites.google.com/corp/view/memory-tasks-suite/home#h.p_2_oxOFZA5QsA).

# Actions

For the 8 Unity-based tasks, the environment provides the following actions:

*   `STRAFE_LEFT_RIGHT`
*   `MOVE_BACK_FORWARD`
*   `LOOK_LEFT_RIGHT`
*   `LOOK_DOWN_UP`

Each action is a `double` scalar, with an inclusive range of `[-1.0, 1.0]`. It
is not compulsory to send a value for each action every step, but note that
actions are "sticky", meaning an action's value will only change when a new
value is provided. For example:

```python
env = dm_memorytasks.load_from_docker(settings)
env.reset()
env.step({'STRAFE_LEFT_RIGHT': -1.0}) # Result: strafe Left.
env.step({'MOVE_BACK_FORWARD': 1.0}) # Result: strafe left & move backward.

env.step({'STRAFE_LEFT_RIGHT': 0.0,
          'MOVE_BACK_FORWARD': 0.0}) # Result: stationary.
```

# Observations

For the 8 Unity-based tasks, the environment provides the following
observations:

*   `RGB_INTERLEAVED`: First person RGB camera observation. The `width` and
    `height` can be adjusted through the `EnvironmentSettings`, but the
    observation will always have a fixed 4:3 aspect ratio.
*   `AvatarPosition`: 3-dimensional world-space position of the agent.
*   `Score`: The agent's cumulative score.

# Configurable environment settings

Required attributes:

*   `seed`: Seed to initialize the environment's RNG.
*   `level_name`: Name of the level to load.

Optional attributes:

*   `width`: Width (in pixels) of the desired RGB observation; defaults to 96.
*   `height`: Height (in pixels) of the desired RGB observation; defaults to 72.
*   `episode_length_seconds`: Maximum episode length (in seconds); defaults
    to 120.
*   `num_action_repeats`: Number of times to step the environment with the
    provided action in calls to `step()`.

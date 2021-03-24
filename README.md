# `dm_memorytasks`: DeepMind Memory Task Suite

The *DeepMind Memory Task Suite* is a set of 13 diverse machine-learning tasks
that require memory to solve. They are constructed to let us evaluate
generalization performance on a memory-specific holdout set.

The 8 tasks in this repo are [Unity-based](http://unity3d.com/). Besides these,
there are 4 tasks in the overall Memory Task Suite that are modifications of
[PsychLab](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/psychlab)
tasks, and 1 that is a modification of a
[DeepMind Lab](https://github.com/deepmind/lab) level.

**NOTE:** The 5 other tasks in the Suite are in Psychlab and DeepMind Lab, not
Unity. Psychlab is part of DeepMind Lab. DeepMind Lab has a separate set of
installation [instructions](https://github.com/deepmind/lab).

## Overview

The 8 Unity-based tasks are provided through a pre-packaged
[Docker container](http://www.docker.com).

The `dm_memorytasks` package consists of support code to run these Docker
containers. You interact with the task environment via a
[`dm_env`](http://www.github.com/deepmind/dm_env) Python interface.

Please see the [documentation](docs/index.md) for more detailed information on
the available tasks, actions and observations.

## Requirements

`dm_memorytasks` requires [Docker](https://www.docker.com),
[Python](https://www.python.org/) 3.6.1 or later and a x86-64 CPU with SSE4.2
support. We do not attempt to maintain a working version for Python 2.

Note: We recommend using
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to
mitigate conflicts with your system's Python environment.

Download and install Docker:

*   For Linux, install [Docker-CE](https://docs.docker.com/install/)
*   Install Docker Desktop for
    [OSX](https://docs.docker.com/docker-for-mac/install/) or
    [Windows](https://docs.docker.com/docker-for-windows/install/).

## Installation

`dm_memorytasks` can be installed from
[PyPi](https://pypi.org/project/dm-memorytasks/) using `pip`:

```bash
$ pip install dm-memorytasks
```

To also install the dependencies for the `examples/`, install with:

```bash
$ pip install dm-memorytasks[examples]
```

Alternatively, you can install `dm_memorytasks` by cloning a local copy of our
GitHub repository:

```bash
$ git clone https://github.com/deepmind/dm_memorytasks.git
$ pip install ./dm_memorytasks
```

## Usage

Once `dm_memorytasks` is installed, to instantiate a `dm_env` instance run the
following:

```python
import dm_memorytasks

settings = dm_memorytasks.EnvironmentSettings(seed=123, level_name='spot_diff_train')
env = dm_memorytasks.load_from_docker(settings)
```

## Citing

If you use `dm_memorytasks` in your work, please cite the accompanying paper:

```bibtex
@inproceedings{fortunato2019generalization,
        title={Generalization of Reinforcement Learners with Working and Episodic Memory},
        author={Fortunato, Meire and
                Tan, Melissa and
                Faulkner, Ryan and
                Hansen, Steven and
                Badia, Adri{\`a} Puigdom{\`e}nech and
                Buttimore, Gavin and
                Deck, Charles and
                Leibo, Joel Z and
                Blundell, Charles},
        booktitle={Advances in Neural Information Processing Systems},
        pages={12448--12457},
        year={2019},
}
```

## Notice

This is not an officially supported Google product.

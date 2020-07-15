# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Python utility functions for loading DeepMind Memory Tasks."""

import codecs
import collections
import json
import os
import re
import subprocess
import time
import typing

from absl import logging
import dm_env
import docker
import grpc
import numpy as np
import portpicker

from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_utils

# Maximum number of times to attempt gRPC connection.
_MAX_CONNECTION_ATTEMPTS = 10

# Port to expect the docker environment to internally listen on.
_DOCKER_INTERNAL_GRPC_PORT = 10000

_DEFAULT_DOCKER_IMAGE_NAME = 'gcr.io/deepmind-environments/dm_memorytasks:v1.0.1'

_MEMORY_TASK_OBSERVATIONS = ['RGB_INTERLEAVED', 'AvatarPosition', 'Score']

MEMORY_TASK_LEVEL_NAMES = frozenset((
    'invisible_goal_empty_arena_extrapolate',
    'invisible_goal_empty_arena_holdout_extrapolate',
    'invisible_goal_empty_arena_holdout_interpolate',
    'invisible_goal_empty_arena_holdout_large',
    'invisible_goal_empty_arena_holdout_small',
    'invisible_goal_empty_arena_interpolate',
    'invisible_goal_empty_arena_train',
    'invisible_goal_with_buildings_extrapolate',
    'invisible_goal_with_buildings_holdout_extrapolate',
    'invisible_goal_with_buildings_holdout_interpolate',
    'invisible_goal_with_buildings_holdout_large',
    'invisible_goal_with_buildings_holdout_small',
    'invisible_goal_with_buildings_interpolate',
    'invisible_goal_with_buildings_train',
    'spot_diff_extrapolate',
    'spot_diff_holdout_extrapolate',
    'spot_diff_holdout_interpolate',
    'spot_diff_holdout_large',
    'spot_diff_holdout_small',
    'spot_diff_interpolate',
    'spot_diff_motion_extrapolate',
    'spot_diff_motion_holdout_extrapolate',
    'spot_diff_motion_holdout_interpolate',
    'spot_diff_motion_holdout_large',
    'spot_diff_motion_holdout_small',
    'spot_diff_motion_interpolate',
    'spot_diff_motion_train',
    'spot_diff_multi_extrapolate',
    'spot_diff_multi_holdout_extrapolate',
    'spot_diff_multi_holdout_interpolate',
    'spot_diff_multi_holdout_large',
    'spot_diff_multi_holdout_small',
    'spot_diff_multi_interpolate',
    'spot_diff_multi_train',
    'spot_diff_passive_extrapolate',
    'spot_diff_passive_holdout_extrapolate',
    'spot_diff_passive_holdout_interpolate',
    'spot_diff_passive_holdout_large',
    'spot_diff_passive_holdout_small',
    'spot_diff_passive_interpolate',
    'spot_diff_passive_train',
    'spot_diff_train',
    'transitive_inference_extrapolate',
    'transitive_inference_holdout_extrapolate',
    'transitive_inference_holdout_interpolate',
    'transitive_inference_holdout_large',
    'transitive_inference_holdout_small',
    'transitive_inference_interpolate',
    'transitive_inference_train_large',
    'transitive_inference_train_small',
    'visible_goal_with_buildings_extrapolate',
    'visible_goal_with_buildings_holdout_extrapolate',
    'visible_goal_with_buildings_holdout_interpolate',
    'visible_goal_with_buildings_holdout_large',
    'visible_goal_with_buildings_holdout_small',
    'visible_goal_with_buildings_interpolate',
    'visible_goal_with_buildings_train',
))

_ConnectionDetails = collections.namedtuple('_ConnectionDetails',
                                            ['channel', 'connection', 'specs'])


class _MemoryTasksEnv(dm_env_adaptor.DmEnvAdaptor):
  """An implementation of dm_env_rpc.DmEnvAdaptor for memory tasks."""

  def __init__(self, connection_details, requested_observations,
               num_action_repeats):
    super(_MemoryTasksEnv,
          self).__init__(connection_details.connection,
                         connection_details.specs, requested_observations)
    self._channel = connection_details.channel
    self._num_action_repeats = num_action_repeats

  def close(self):
    super(_MemoryTasksEnv, self).close()
    self._channel.close()

  def step(self, action):
    """Implementation of dm_env.step that supports repeated actions."""

    timestep = None
    discount = None
    reward = None
    for _ in range(self._num_action_repeats):
      next_timestep = super(_MemoryTasksEnv, self).step(action)

      # Accumulate reward per timestep.
      if next_timestep.reward is not None:
        reward = (reward or 0.) + next_timestep.reward

      # Calculate the product for discount.
      if next_timestep.discount is not None:
        discount = discount if discount else []
        discount.append(next_timestep.discount)

      timestep = dm_env.TimeStep(next_timestep.step_type, reward,
                                 # Note: np.product(None) returns None.
                                 np.product(discount),
                                 next_timestep.observation)

      if timestep.last():
        return timestep

    return timestep


class _MemoryTasksContainerEnv(_MemoryTasksEnv):
  """An implementation of _MemoryTasksEnv.

    Ensures that the provided Docker container is closed on exit.
  """

  def __init__(self, connection_details, requested_observations,
               num_action_repeats, container):
    super(_MemoryTasksContainerEnv,
          self).__init__(connection_details, requested_observations,
                         num_action_repeats)
    self._container = container

  def close(self):
    super(_MemoryTasksContainerEnv, self).close()
    try:
      self._container.kill()
    except docker.errors.NotFound:
      pass  # Ignore, container has already been closed.


class _MemoryTasksProcessEnv(_MemoryTasksEnv):
  """An implementation of _MemoryTasksEnv.

    Ensure that the provided running process is closed on exit.
  """

  def __init__(self, connection_details, requested_observations,
               num_action_repeats, process):
    super(_MemoryTasksProcessEnv,
          self).__init__(connection_details, requested_observations,
                         num_action_repeats)
    self._process = process

  def close(self):
    super(_MemoryTasksProcessEnv, self).close()
    self._process.terminate()
    self._process.wait()


def _check_grpc_channel_ready(channel):
  """Helper function to check the gRPC channel is ready N times."""
  for _ in range(_MAX_CONNECTION_ATTEMPTS - 1):
    try:
      return grpc.channel_ready_future(channel).result(timeout=1)
    except grpc.FutureTimeoutError:
      pass
  return grpc.channel_ready_future(channel).result(timeout=1)


def _can_send_message(connection):
  """Returns if `connection` is healthy and able to process requests."""
  try:
    # This should return a response with an error unless the server isn't yet
    # receiving requests.
    connection.send(dm_env_rpc_pb2.StepRequest())
  except error.DmEnvRpcError:
    return True
  except grpc.RpcError:
    return False


def _create_channel_and_connection(port):
  """Returns a tuple of `(channel, connection)`."""
  for _ in range(_MAX_CONNECTION_ATTEMPTS):
    channel = grpc.secure_channel('localhost:{}'.format(port),
                                  grpc.local_channel_credentials())
    _check_grpc_channel_ready(channel)
    connection = dm_env_rpc_connection.Connection(channel)
    if _can_send_message(connection):
      break
    else:
      # A gRPC server running within Docker sometimes reports that the channel
      # is ready but transitively returns an error (status code 14) on first
      # use.  Giving the server some time to breath and retrying often fixes the
      # problem.
      connection.close()
      channel.close()
      time.sleep(1.0)

  return channel, connection


def _parse_exception_message(message):
  """Returns a human-readable version of a dm_env_rpc json error message."""
  try:
    match = re.match(r'^message\:\ \"(.*)\"$', message)
    json_data = codecs.decode(match.group(1), 'unicode-escape')
    parsed_json_data = json.loads(json_data)
    return ValueError(json.dumps(parsed_json_data, indent=4))
  except:  # pylint: disable=bare-except
    return message


def _wrap_send(send):
  """Wraps `send` in order to reformat exceptions."""
  try:
    return send()
  except ValueError as e:
    e.args = [_parse_exception_message(e.args[0])]
    raise


def _connect_to_environment(port, settings):
  """Helper function for connecting to a running dm_memorytask environment."""
  if settings.level_name not in MEMORY_TASK_LEVEL_NAMES:
    raise ValueError(
        'Level named "{}" is not a valid dm_memorytask level.'.format(
            settings.level_name))
  channel, connection = _create_channel_and_connection(port)
  original_send = connection.send
  connection.send = lambda request: _wrap_send(lambda: original_send(request))
  world_name = connection.send(
      dm_env_rpc_pb2.CreateWorldRequest(
          settings={
              'seed': tensor_utils.pack_tensor(settings.seed),
              'episodeId': tensor_utils.pack_tensor(0),
              'levelName': tensor_utils.pack_tensor(settings.level_name),
          })).world_name
  join_world_settings = {
      'width':
          tensor_utils.pack_tensor(settings.width),
      'height':
          tensor_utils.pack_tensor(settings.height),
      'EpisodeLengthSeconds':
          tensor_utils.pack_tensor(settings.episode_length_seconds),
  }
  specs = connection.send(
      dm_env_rpc_pb2.JoinWorldRequest(
          world_name=world_name, settings=join_world_settings)).specs
  return _ConnectionDetails(channel=channel, connection=connection, specs=specs)


class EnvironmentSettings(typing.NamedTuple):
  """Collection of settings used to start a specific Memory task.

    Required attributes:
      seed: Seed to initialize the environment's RNG.
      level_name: Name of the level to load.
    Optional attributes:
      width: Width (in pixels) of the desired RGB observation; defaults to 96.
      height: Height (in pixels) of the desired RGB observation; defaults to 72.
      episode_length_seconds: Maximum episode length (in seconds); defaults to
        120.
      num_action_repeats: Number of times to step the environment with the
        provided action in calls to `step()`.
  """
  seed: int
  level_name: str
  width: int = 96
  height: int = 72
  episode_length_seconds: float = 120.0
  num_action_repeats: int = 1


def _validate_environment_settings(settings):
  """Helper function to validate the provided environment settings."""
  if settings.episode_length_seconds <= 0.0:
    raise ValueError('episode_length_seconds must have a positive value.')
  if settings.num_action_repeats <= 0:
    raise ValueError('num_action_repeats must have a positive value.')
  if settings.width <= 0 or settings.height <= 0:
    raise ValueError('width and height must have a positive value.')


def load_from_disk(path, settings):
  """Load Memory Tasks from disk.

  Args:
    path: Directory containing dm_memorytasks environment.
    settings: EnvironmentSettings required to start the environment.

  Returns:
    An implementation of dm_env.Environment.

  Raises:
    RuntimeError: If unable to start environment process.
  """
  _validate_environment_settings(settings)

  executable_path = os.path.join(path, 'Linux64Player')
  libosmesa_path = os.path.join(path, 'external_libosmesa_llvmpipe.so')
  if not os.path.exists(executable_path) or not os.path.exists(libosmesa_path):
    raise RuntimeError(
        'Cannot find dm_memorytasks executable or dependent files at path: {}'
        .format(path))

  port = portpicker.pick_unused_port()

  process_flags = [
      executable_path,
      # Unity command-line flags.
      '-logfile',
      '-batchmode',
      '-noaudio',
      # Other command-line flags.
      '--logtostderr',
      '--server_type=GRPC',
      '--uri_address=[::]:{}'.format(port),
  ]

  os.environ.update({
      'UNITY_RENDERER': 'software',
      'UNITY_OSMESA_PATH': libosmesa_path,
  })

  process = subprocess.Popen(
      process_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  if process.poll() is not None:
    raise RuntimeError('Failed to start dm_memorytasks process correctly.')

  return _MemoryTasksProcessEnv(
      _connect_to_environment(port, settings), _MEMORY_TASK_OBSERVATIONS,
      settings.num_action_repeats, process)


def load_from_docker(settings, name=None):
  """Load Memory Tasks from docker container.

  Args:
    settings: EnvironmentSettings required to start the environment.
    name: Optional name of Docker image that contains the dm_memorytasks
      environment. If left unset, uses the dm_memorytasks default name.

  Returns:
    An implementation of dm_env.Environment
  """
  _validate_environment_settings(settings)

  name = name or _DEFAULT_DOCKER_IMAGE_NAME
  client = docker.from_env()

  port = portpicker.pick_unused_port()

  try:
    client.images.get(name)
  except docker.errors.ImageNotFound:
    logging.info('Downloading docker image "%s"...', name)
    client.images.pull(name)
    logging.info('Download finished.')

  container = client.containers.run(
      name,
      auto_remove=True,
      detach=True,
      ports={_DOCKER_INTERNAL_GRPC_PORT: port})

  return _MemoryTasksContainerEnv(
      _connect_to_environment(port, settings), _MEMORY_TASK_OBSERVATIONS,
      settings.num_action_repeats, container)

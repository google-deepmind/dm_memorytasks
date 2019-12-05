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
"""Example human agent for interacting with DeepMind Memory Tasks."""

from absl import app
from absl import flags
from absl import logging
import dm_memorytasks
import numpy as np
import pygame

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'screen_size', [640, 480],
    'Screen width/height in pixels. Scales the environment RGB observations to '
    'fit the screen size.')

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Memory Tasks. '
    'If None, uses the default dm_memorytask name')

flags.DEFINE_integer('seed', 123, 'Environment seed.')
flags.DEFINE_string('level_name', 'spot_diff_extrapolate',
                    'Name of memory task to run.')

_FRAMES_PER_SECOND = 30

_KEYS_TO_ACTION = {
    pygame.K_w: {'MOVE_BACK_FORWARD': 1},
    pygame.K_s: {'MOVE_BACK_FORWARD': -1},
    pygame.K_a: {'STRAFE_LEFT_RIGHT': -1},
    pygame.K_d: {'STRAFE_LEFT_RIGHT': 1},
    pygame.K_UP: {'LOOK_DOWN_UP': -1},
    pygame.K_DOWN: {'LOOK_DOWN_UP': 1},
    pygame.K_LEFT: {'LOOK_LEFT_RIGHT': -1},
    pygame.K_RIGHT: {'LOOK_LEFT_RIGHT': 1},
}  # pyformat: disable
_NO_ACTION = {
    'MOVE_BACK_FORWARD': 0,
    'STRAFE_LEFT_RIGHT': 0,
    'LOOK_LEFT_RIGHT': 0,
    'LOOK_DOWN_UP': 0,
}


def main(_):
  pygame.init()
  pygame.display.set_caption('Memory Tasks Human Agent')

  env_settings = dm_memorytasks.EnvironmentSettings(
      seed=FLAGS.seed, level_name=FLAGS.level_name)
  with dm_memorytasks.load_from_docker(name=FLAGS.docker_image_name,
                                       settings=env_settings) as env:
    screen = pygame.display.set_mode(
        (int(FLAGS.screen_size[0]), int(FLAGS.screen_size[1])))

    rgb_spec = env.observation_spec()['RGB_INTERLEAVED']
    surface = pygame.Surface((rgb_spec.shape[1], rgb_spec.shape[0]))

    actions = _NO_ACTION
    score = 0
    clock = pygame.time.Clock()
    while True:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            return
          key_actions = _KEYS_TO_ACTION.get(event.key, {})
          for name, action in key_actions.items():
            actions[name] += action
        elif event.type == pygame.KEYUP:
          key_actions = _KEYS_TO_ACTION.get(event.key, {})
          for name, action in key_actions.items():
            actions[name] -= action

      timestep = env.step(actions)
      frame = np.swapaxes(timestep.observation['RGB_INTERLEAVED'], 0, 1)
      pygame.surfarray.blit_array(surface, frame)
      pygame.transform.smoothscale(surface, screen.get_size(), screen)

      pygame.display.update()

      if timestep.reward:
        score += timestep.reward
        logging.info('Total score: %1.1f, reward: %1.1f', score,
                     timestep.reward)
      clock.tick(_FRAMES_PER_SECOND)


if __name__ == '__main__':
  app.run(main)

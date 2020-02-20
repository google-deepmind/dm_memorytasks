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
"""Tests for dm_memorytasks.load_from_docker."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dm_env import test_utils
import dm_memorytasks

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Memory Tasks. '
    'If None, uses the default dm_memorytask name')


class LoadFromDockerTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return dm_memorytasks.load_from_docker(
        name=FLAGS.docker_image_name,
        settings=dm_memorytasks.EnvironmentSettings(
            seed=123, level_name='spot_diff_extrapolate'))


class MemoryTaskTest(parameterized.TestCase):

  @parameterized.parameters(dm_memorytasks.MEMORY_TASK_LEVEL_NAMES)
  def test_load_level(self, level_name):
    self.assertIsNotNone(
        dm_memorytasks.load_from_docker(
            name=FLAGS.docker_image_name,
            settings=dm_memorytasks.EnvironmentSettings(
                seed=123, level_name=level_name)))


if __name__ == '__main__':
  absltest.main()

# Copyright 2020, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import numpy as np

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import models
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData


class TrainedAttackerTest(absltest.TestCase):

  def test_base_attacker_train_and_predict(self):
    base_attacker = models.TrainedAttacker()
    self.assertRaises(NotImplementedError, base_attacker.train_model, [], [])
    self.assertRaises(AssertionError, base_attacker.predict, [])

  def test_predict_before_training(self):
    lr_attacker = models.LogisticRegressionAttacker()
    self.assertRaises(AssertionError, lr_attacker.predict, [])

  def test_create_attacker_data_loss_only(self):
    attack_input = AttackInputData(
        loss_train=np.array([1, 3]), loss_test=np.array([2, 4]))
    attacker_data = models.create_attacker_data(attack_input, 2)
    self.assertLen(attacker_data.features_all, 4)

  def test_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16]))
    attacker_data = models.create_attacker_data(attack_input, balance=False)
    self.assertLen(attacker_data.features_all, 5)
    self.assertLen(attacker_data.fold_indices, 5)
    self.assertEmpty(attacker_data.left_out_indices)

  def test_unbalanced_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16]))
    attacker_data = models.create_attacker_data(attack_input, balance=True)
    self.assertLen(attacker_data.features_all, 5)
    self.assertLen(attacker_data.fold_indices, 4)
    self.assertLen(attacker_data.left_out_indices, 1)
    self.assertIn(attacker_data.left_out_indices[0], [0, 1, 2])

  def test_balanced_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15], [17, 18]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16, 19]))
    attacker_data = models.create_attacker_data(attack_input)
    self.assertLen(attacker_data.features_all, 6)
    self.assertLen(attacker_data.fold_indices, 6)
    self.assertEmpty(attacker_data.left_out_indices)


if __name__ == '__main__':
  absltest.main()

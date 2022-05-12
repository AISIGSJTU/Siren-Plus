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
"""Code that runs membership inference attacks based on the model outputs.

This file belongs to the new API for membership inference attacks. This file
will be renamed to membership_inference_attack.py after the old API is removed.
"""

from typing import Iterable

import numpy as np
from sklearn import metrics
from sklearn import model_selection

from utils.privacy_tests.membership_inference_attack import models
from utils.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from utils.privacy_tests.membership_inference_attack.data_structures import AttackResults
from utils.privacy_tests.membership_inference_attack.data_structures import AttackType
from utils.privacy_tests.membership_inference_attack.data_structures import DataSize
from utils.privacy_tests.membership_inference_attack.data_structures import MembershipProbabilityResults
from utils.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from utils.privacy_tests.membership_inference_attack.data_structures import RocCurve
from utils.privacy_tests.membership_inference_attack.data_structures import SingleAttackResult
from utils.privacy_tests.membership_inference_attack.data_structures import SingleMembershipProbabilityResult
from utils.privacy_tests.membership_inference_attack.data_structures import SingleSliceSpec
from utils.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from utils.privacy_tests.membership_inference_attack.dataset_slicing import get_single_slice_specs
from utils.privacy_tests.membership_inference_attack.dataset_slicing import get_slice


def _get_slice_spec(data: AttackInputData) -> SingleSliceSpec:
  if hasattr(data, 'slice_spec'):
    return data.slice_spec
  return SingleSliceSpec()


# TODO(b/220394926): Allow users to specify their own attack models.
def _run_trained_attack(attack_input: AttackInputData,
                        attack_type: AttackType,
                        balance_attacker_training: bool = True,
                        cross_validation_folds: int = 2):
  """Classification attack done by ML models."""
  prepared_attacker_data = models.create_attacker_data(
      attack_input, balance=balance_attacker_training)
  indices = prepared_attacker_data.fold_indices
  left_out_indices = prepared_attacker_data.left_out_indices
  features = prepared_attacker_data.features_all
  labels = prepared_attacker_data.labels_all

  # We are going to train multiple models on disjoint subsets of the data
  # (`features`, `labels`), so we can get the membership scores of all samples,
  # and each example gets its score assigned only once.
  # An alternative implementation is to train multiple models on overlapping
  # subsets of the data, and take an average to get the score for each sample.
  # `scores` will record the membership score of each sample, initialized to nan
  scores = np.full(features.shape[0], np.nan)

  # We use StratifiedKFold to create disjoint subsets of samples. Notice that
  # the index it returns is with respect to the samples shuffled with `indices`.
  kf = model_selection.StratifiedKFold(cross_validation_folds, shuffle=False)
  for train_indices_in_shuffled, test_indices_in_shuffled in kf.split(
      features[indices], labels[indices]):
    # `train_indices_in_shuffled` is with respect to the data shuffled with
    # `indices`. We convert it to `train_indices` to work with the original
    # data (`features` and 'labels').
    train_indices = indices[train_indices_in_shuffled]
    test_indices = indices[test_indices_in_shuffled]
    # Make sure one sample only got score predicted once
    assert np.all(np.isnan(scores[test_indices]))

    attacker = models.create_attacker(attack_type)
    attacker.train_model(features[train_indices], labels[train_indices])
    scores[test_indices] = attacker.predict(features[test_indices])

  # Predict the left out with the last attacker
  if left_out_indices.size:
    assert np.all(np.isnan(scores[left_out_indices]))
    scores[left_out_indices] = attacker.predict(features[left_out_indices])
  assert not np.any(np.isnan(scores))

  # Generate ROC curves with scores.
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
  with open('output/fpr.txt', 'w') as outf:
    # outf.write('FPR\n')
    for index in range(len(fpr)):
      outf.write(str(fpr[index])+'\n')
  with open('output/tpr.txt', 'w') as outf:
    # outf.write('TPR\n')
    for index in range(len(fpr)):
      outf.write(str(tpr[index])+'\n')
  with open('output/thresholds.txt', 'w') as outf:
    # outf.write('Thresholds\n')
    for index in range(len(fpr)):
      outf.write(str(thresholds[index])+'\n')
    
  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  in_train_indices = (labels == 0)
  return SingleAttackResult(
      slice_spec=_get_slice_spec(attack_input),
      data_size=prepared_attacker_data.data_size,
      attack_type=attack_type,
      membership_scores_train=scores[in_train_indices],
      membership_scores_test=scores[~in_train_indices],
      roc_curve=roc_curve)


def _run_threshold_attack(attack_input: AttackInputData):
  """Runs a threshold attack on loss."""
  ntrain, ntest = attack_input.get_train_size(), attack_input.get_test_size()
  loss_train = attack_input.get_loss_train()
  loss_test = attack_input.get_loss_test()
  if loss_train is None or loss_test is None:
    raise ValueError('Not possible to run threshold attack without losses.')
  fpr, tpr, thresholds = metrics.roc_curve(
      np.concatenate((np.zeros(ntrain), np.ones(ntest))),
      np.concatenate((loss_train, loss_test)))
  
  with open('output/fpr.txt', 'w') as outf:
    # outf.write('FPR\n')
    for index in range(len(fpr)):
      outf.write(str(fpr[index])+'\n')
  with open('output/tpr.txt', 'w') as outf:
    # outf.write('TPR\n')
    for index in range(len(fpr)):
      outf.write(str(tpr[index])+'\n')
  with open('output/thresholds.txt', 'w') as outf:
    # outf.write('Thresholds\n')
    for index in range(len(fpr)):
      outf.write(str(thresholds[index])+'\n')

  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  return SingleAttackResult(
      slice_spec=_get_slice_spec(attack_input),
      data_size=DataSize(ntrain=ntrain, ntest=ntest),
      attack_type=AttackType.THRESHOLD_ATTACK,
      membership_scores_train=attack_input.get_loss_train(),
      membership_scores_test=attack_input.get_loss_test(),
      roc_curve=roc_curve)


def _run_threshold_entropy_attack(attack_input: AttackInputData):
  ntrain, ntest = attack_input.get_train_size(), attack_input.get_test_size()
  fpr, tpr, thresholds = metrics.roc_curve(
      np.concatenate((np.zeros(ntrain), np.ones(ntest))),
      np.concatenate(
          (attack_input.get_entropy_train(), attack_input.get_entropy_test())))
  
  with open('output/fpr.txt', 'w') as outf:
    # outf.write('FPR\n')
    for index in range(len(fpr)):
      outf.write(str(fpr[index])+'\n')
  with open('output/tpr.txt', 'w') as outf:
    # outf.write('TPR\n')
    for index in range(len(fpr)):
      outf.write(str(tpr[index])+'\n')
  with open('output/thresholds.txt', 'w') as outf:
    # outf.write('Thresholds\n')
    for index in range(len(fpr)):
      outf.write(str(thresholds[index])+'\n')

  roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)

  return SingleAttackResult(
      slice_spec=_get_slice_spec(attack_input),
      data_size=DataSize(ntrain=ntrain, ntest=ntest),
      attack_type=AttackType.THRESHOLD_ENTROPY_ATTACK,
      membership_scores_train=-attack_input.get_entropy_train(),
      membership_scores_test=-attack_input.get_entropy_test(),
      roc_curve=roc_curve)


def _run_attack(attack_input: AttackInputData,
                attack_type: AttackType,
                balance_attacker_training: bool = True,
                min_num_samples: int = 1):
  """Runs membership inference attacks for specified input and type.

  Args:
    attack_input: input data for running an attack
    attack_type: the attack to run
    balance_attacker_training: Whether the training and test sets for the
      membership inference attacker should have a balanced (roughly equal)
      number of samples from the training and test sets used to develop the
      model under attack.
    min_num_samples: minimum number of examples in either training or test data.

  Returns:
    the attack result.
  """
  attack_input.validate()
  if min(attack_input.get_train_size(),
         attack_input.get_test_size()) < min_num_samples:
    return None

  if attack_type.is_trained_attack:
    return _run_trained_attack(attack_input, attack_type,
                               balance_attacker_training)
  if attack_type == AttackType.THRESHOLD_ENTROPY_ATTACK:
    return _run_threshold_entropy_attack(attack_input)
  return _run_threshold_attack(attack_input)


def run_attacks(attack_input: AttackInputData,
                slicing_spec: SlicingSpec = None,
                attack_types: Iterable[AttackType] = (
                    AttackType.THRESHOLD_ATTACK,),
                privacy_report_metadata: PrivacyReportMetadata = None,
                balance_attacker_training: bool = True,
                min_num_samples: int = 1) -> AttackResults:
  """Runs membership inference attacks on a classification model.

  It runs attacks specified by attack_types on each attack_input slice which is
   specified by slicing_spec.

  Args:
    attack_input: input data for running an attack
    slicing_spec: specifies attack_input slices to run attack on
    attack_types: attacks to run
    privacy_report_metadata: the metadata of the model under attack.
    balance_attacker_training: Whether the training and test sets for the
      membership inference attacker should have a balanced (roughly equal)
      number of samples from the training and test sets used to develop the
      model under attack.
    min_num_samples: minimum number of examples in either training or test data.

  Returns:
    the attack result.
  """
  attack_input.validate()
  attack_results = []

  if slicing_spec is None:
    slicing_spec = SlicingSpec(entire_dataset=True)
  num_classes = None
  if slicing_spec.by_class:
    num_classes = attack_input.num_classes
  input_slice_specs = get_single_slice_specs(slicing_spec, num_classes)
  for single_slice_spec in input_slice_specs:
    attack_input_slice = get_slice(attack_input, single_slice_spec)
    for attack_type in attack_types:
      attack_result = _run_attack(attack_input_slice, attack_type,
                                  balance_attacker_training, min_num_samples)
      if attack_result is not None:
        attack_results.append(attack_result)

  privacy_report_metadata = _compute_missing_privacy_report_metadata(
      privacy_report_metadata, attack_input)

  return AttackResults(
      single_attack_results=attack_results,
      privacy_report_metadata=privacy_report_metadata)


def _compute_membership_probability(
    attack_input: AttackInputData,
    num_bins: int = 15) -> SingleMembershipProbabilityResult:
  """Computes each individual point's likelihood of being a member (denoted as privacy risk score in https://arxiv.org/abs/2003.10595).

  For an individual sample, its privacy risk score is computed as the posterior
  probability of being in the training set
  after observing its prediction output by the target machine learning model.

  Args:
    attack_input: input data for compute membership probability
    num_bins: the number of bins used to compute the training/test histogram

  Returns:
    membership probability results
  """

  # Uses the provided loss or entropy. Otherwise computes the loss.
  if attack_input.loss_train is not None and attack_input.loss_test is not None:
    train_values = attack_input.loss_train
    test_values = attack_input.loss_test
  elif attack_input.entropy_train is not None and attack_input.entropy_test is not None:
    train_values = attack_input.entropy_train
    test_values = attack_input.entropy_test
  else:
    train_values = attack_input.get_loss_train()
    test_values = attack_input.get_loss_test()

  # Compute the histogram in the log scale
  small_value = 1e-10
  train_values = np.maximum(train_values, small_value)
  test_values = np.maximum(test_values, small_value)

  min_value = min(train_values.min(), test_values.min())
  max_value = max(train_values.max(), test_values.max())
  bins_hist = np.logspace(
      np.log10(min_value), np.log10(max_value), num_bins + 1)

  train_hist, _ = np.histogram(train_values, bins=bins_hist)
  train_hist = train_hist / (len(train_values) + 0.0)
  train_hist_indices = np.fmin(
      np.digitize(train_values, bins=bins_hist), num_bins) - 1

  test_hist, _ = np.histogram(test_values, bins=bins_hist)
  test_hist = test_hist / (len(test_values) + 0.0)
  test_hist_indices = np.fmin(
      np.digitize(test_values, bins=bins_hist), num_bins) - 1

  combined_hist = train_hist + test_hist
  combined_hist[combined_hist == 0] = small_value
  membership_prob_list = train_hist / (combined_hist + 0.0)
  train_membership_probs = membership_prob_list[train_hist_indices]
  test_membership_probs = membership_prob_list[test_hist_indices]

  return SingleMembershipProbabilityResult(
      slice_spec=_get_slice_spec(attack_input),
      train_membership_probs=train_membership_probs,
      test_membership_probs=test_membership_probs)


def run_membership_probability_analysis(
    attack_input: AttackInputData,
    slicing_spec: SlicingSpec = None) -> MembershipProbabilityResults:
  """Perform membership probability analysis on all given slice types.

  Args:
    attack_input: input data for compute membership probabilities
    slicing_spec: specifies attack_input slices

  Returns:
    the membership probability results.
  """
  attack_input.validate()
  membership_prob_results = []

  if slicing_spec is None:
    slicing_spec = SlicingSpec(entire_dataset=True)
  num_classes = None
  if slicing_spec.by_class:
    num_classes = attack_input.num_classes
  input_slice_specs = get_single_slice_specs(slicing_spec, num_classes)
  for single_slice_spec in input_slice_specs:
    attack_input_slice = get_slice(attack_input, single_slice_spec)
    membership_prob_results.append(
        _compute_membership_probability(attack_input_slice))

  return MembershipProbabilityResults(
      membership_prob_results=membership_prob_results)


def _compute_missing_privacy_report_metadata(
    metadata: PrivacyReportMetadata,
    attack_input: AttackInputData) -> PrivacyReportMetadata:
  """Populates metadata fields if they are missing."""
  if metadata is None:
    metadata = PrivacyReportMetadata()
  if metadata.accuracy_train is None:
    metadata.accuracy_train = _get_accuracy(attack_input.logits_train,
                                            attack_input.labels_train)
  if metadata.accuracy_test is None:
    metadata.accuracy_test = _get_accuracy(attack_input.logits_test,
                                           attack_input.labels_test)
  loss_train = attack_input.get_loss_train()
  loss_test = attack_input.get_loss_test()
  if metadata.loss_train is None and loss_train is not None:
    metadata.loss_train = np.average(loss_train)
  if metadata.loss_test is None and loss_test is not None:
    metadata.loss_test = np.average(loss_test)
  return metadata


def _get_accuracy(logits, labels):
  """Computes the accuracy if it is missing."""
  if logits is None or labels is None:
    return None
  return metrics.accuracy_score(labels, np.argmax(logits, axis=1))

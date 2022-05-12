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
"""Data structures representing attack inputs, configuration, outputs."""

import collections
import dataclasses
import enum
import glob
import os
import pickle
from typing import Any, Callable, Iterable, MutableSequence, Optional, Union

import numpy as np
import pandas as pd
from scipy import special
from sklearn import metrics
import utils.privacy_tests.membership_inference_attack.utils as utils

ENTIRE_DATASET_SLICE_STR = 'Entire dataset'


class SlicingFeature(enum.Enum):
  """Enum with features by which slicing is available."""
  CLASS = 'class'
  PERCENTILE = 'percentile'
  CORRECTLY_CLASSIFIED = 'correctly_classified'


@dataclasses.dataclass
class SingleSliceSpec:
  """Specifies a slice.

  The slice is defined by values in one feature - it might be a single value
  (eg. slice of examples of the specific classification class) or some set of
  values (eg. range of percentiles of the attacked model loss).

  When feature is None, it means that the slice is the entire dataset.
  """
  feature: Optional[SlicingFeature] = None
  value: Optional[Any] = None

  @property
  def entire_dataset(self):
    return self.feature is None

  def __str__(self):
    if self.entire_dataset:
      return ENTIRE_DATASET_SLICE_STR

    if self.feature == SlicingFeature.PERCENTILE:
      return 'Loss percentiles: %d-%d' % self.value

    return '%s=%s' % (self.feature.name, self.value)


@dataclasses.dataclass
class SlicingSpec:
  """Specification of a slicing procedure.

  Each variable which is set specifies a slicing by different dimension.
  """

  # When is set to true, one of the slices is the whole dataset.
  entire_dataset: bool = True

  # Used in classification tasks for slicing by classes. It is assumed that
  # classes are integers 0, 1, ... number of classes. When true one slice per
  # each class is generated.
  by_class: Union[bool, Iterable[int], int] = False

  # if true, it generates 10 slices for percentiles of the loss - 0-10%, 10-20%,
  # ... 90-100%.
  by_percentiles: bool = False

  # When true, a slice for correctly classifed and a slice for misclassifed
  # examples will be generated.
  by_classification_correctness: bool = False

  def __str__(self):
    """Only keeps the True values."""
    result = ['SlicingSpec(']
    if self.entire_dataset:
      result.append(' Entire dataset,')
    if self.by_class:
      if isinstance(self.by_class, Iterable):
        result.append(' Into classes %s,' % self.by_class)
      elif isinstance(self.by_class, int):
        result.append(' Up to class %d,' % self.by_class)
      else:
        result.append(' By classes,')
    if self.by_percentiles:
      result.append(' By percentiles,')
    if self.by_classification_correctness:
      result.append(' By classification correctness,')
    result.append(')')
    return '\n'.join(result)


class AttackType(enum.Enum):
  """An enum define attack types."""
  LOGISTIC_REGRESSION = 'lr'
  MULTI_LAYERED_PERCEPTRON = 'mlp'
  RANDOM_FOREST = 'rf'
  K_NEAREST_NEIGHBORS = 'knn'
  THRESHOLD_ATTACK = 'threshold'
  THRESHOLD_ENTROPY_ATTACK = 'threshold-entropy'

  @property
  def is_trained_attack(self):
    """Returns whether this type of attack requires training a model."""
    return (self != AttackType.THRESHOLD_ATTACK) and (
        self != AttackType.THRESHOLD_ENTROPY_ATTACK)

  def __str__(self):
    """Returns LOGISTIC_REGRESSION instead of AttackType.LOGISTIC_REGRESSION."""
    return '%s' % self.name


class PrivacyMetric(enum.Enum):
  """An enum for the supported privacy risk metrics."""
  AUC = 'AUC'
  ATTACKER_ADVANTAGE = 'Attacker advantage'

  def __str__(self):
    """Returns 'AUC' instead of PrivacyMetric.AUC."""
    return '%s' % self.value


def _is_integer_type_array(a):
  return np.issubdtype(a.dtype, np.integer)


def _is_last_dim_equal(arr1, arr1_name, arr2, arr2_name):
  """Checks whether the last dimension of the arrays is the same."""
  if arr1 is not None and arr2 is not None and arr1.shape[-1] != arr2.shape[-1]:
    raise ValueError('%s and %s should have the same number of features.' %
                     (arr1_name, arr2_name))


def _is_array_one_dimensional(arr, arr_name):
  """Checks whether the array is one dimensional."""
  if arr is not None and len(arr.shape) != 1:
    raise ValueError('%s should be a one dimensional numpy array.' % arr_name)


def _is_np_array(arr, arr_name):
  """Checks whether array is a numpy array."""
  if arr is not None and not isinstance(arr, np.ndarray):
    raise ValueError('%s should be a numpy array.' % arr_name)


def _log_value(probs, small_value=1e-30):
  """Compute the log value on the probability. Clip probabilities close to 0."""
  return -np.log(np.maximum(probs, small_value))


class LossFunction(enum.Enum):
  """An enum that defines loss function to use in `AttackInputData`."""
  CROSS_ENTROPY = 'cross_entropy'
  SQUARED = 'squared'


@dataclasses.dataclass
class AttackInputData:
  """Input data for running an attack.

  This includes only the data, and not configuration.
  """

  logits_train: Optional[np.ndarray] = None
  logits_test: Optional[np.ndarray] = None

  # Predicted probabilities for each class. They can be derived from logits,
  # so they can be set only if logits are not explicitly provided.
  probs_train: Optional[np.ndarray] = None
  probs_test: Optional[np.ndarray] = None

  # Contains ground-truth classes. Classes are assumed to be integers starting
  # from 0.
  labels_train: Optional[np.ndarray] = None
  labels_test: Optional[np.ndarray] = None

  # Explicitly specified loss. If provided, this is used instead of deriving
  # loss from logits and labels
  loss_train: Optional[np.ndarray] = None
  loss_test: Optional[np.ndarray] = None

  # Explicitly specified prediction entropy. If provided, this is used instead
  # of deriving entropy from logits and labels
  # (https://arxiv.org/pdf/2003.10595.pdf by Song and Mittal).
  entropy_train: Optional[np.ndarray] = None
  entropy_test: Optional[np.ndarray] = None

  # If loss is not explicitly specified, this function will be used to derive
  # loss from logits and labels. It can be a pre-defined `LossFunction`.
  # If a callable is provided, it should take in two argument, the 1st is
  # labels, the 2nd is logits or probs.
  loss_function: Union[Callable[[np.ndarray, np.ndarray], np.ndarray],
                       LossFunction] = LossFunction.CROSS_ENTROPY
  # Whether `loss_function` will be called with logits or probs. If not set
  # (None), will decide by availablity of logits and probs and logits is
  # preferred when both are available.
  loss_function_using_logits: Optional[bool] = None

  @property
  def num_classes(self):
    if self.labels_train is None or self.labels_test is None:
      raise ValueError(
          'Can\'t identify the number of classes as no labels were provided. '
          'Please set labels_train and labels_test')
    return int(max(np.max(self.labels_train), np.max(self.labels_test))) + 1

  @property
  def logits_or_probs_train(self):
    """Returns train logits or probs whatever is not None."""
    if self.logits_train is not None:
      return self.logits_train
    return self.probs_train

  @property
  def logits_or_probs_test(self):
    """Returns test logits or probs whatever is not None."""
    if self.logits_test is not None:
      return self.logits_test
    return self.probs_test

  @staticmethod
  def _get_entropy(logits: np.ndarray, true_labels: np.ndarray):
    """Computes the prediction entropy (by Song and Mittal)."""
    if (np.absolute(np.sum(logits, axis=1) - 1) <= 1e-3).all():
      probs = logits
    else:
      # Using softmax to compute probability from logits.
      probs = special.softmax(logits, axis=1)
    if true_labels is None:
      # When not given ground truth label, we compute the
      # normal prediction entropy.
      # See the Equation (7) in https://arxiv.org/pdf/2003.10595.pdf
      return np.sum(np.multiply(probs, _log_value(probs)), axis=1)
    else:
      # When given the ground truth label, we compute the
      # modified prediction entropy.
      # See the Equation (8) in https://arxiv.org/pdf/2003.10595.pdf
      log_probs = _log_value(probs)
      reverse_probs = 1 - probs
      log_reverse_probs = _log_value(reverse_probs)
      modified_probs = np.copy(probs)
      modified_probs[range(true_labels.size),
                     true_labels] = reverse_probs[range(true_labels.size),
                                                  true_labels]
      modified_log_probs = np.copy(log_reverse_probs)
      modified_log_probs[range(true_labels.size),
                         true_labels] = log_probs[range(true_labels.size),
                                                  true_labels]
      return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

  @staticmethod
  def _get_loss(
      loss: Optional[np.ndarray], labels: Optional[np.ndarray],
      logits: Optional[np.ndarray], probs: Optional[np.ndarray],
      loss_function: Union[Callable[[np.ndarray, np.ndarray], np.ndarray],
                           LossFunction],
      loss_function_using_logits: Optional[bool]) -> Optional[np.ndarray]:
    """Calculates (if needed) losses.

    Args:
      loss: the loss of each example.
      labels: the scalar label of each example.
      logits: the logits vector of each example.
      probs: the probability vector of each example.
      loss_function: if `loss` is not available, `labels` and one of `logits`
        and `probs` are available, we will use this function to compute loss. It
        is supposed to take in (label, logits / probs) as input.
      loss_function_using_logits: if `loss_function` expects `logits` or
        `probs`.

    Returns:
      Loss (or None if neither the loss nor the labels are present).
    """
    if loss is not None:
      return loss
    if labels is None or (logits is None and probs is None):
      return None
    if loss_function_using_logits and logits is None:
      raise ValueError('We need logits to compute loss, but it is set to None.')
    if not loss_function_using_logits and probs is None:
      raise ValueError('We need probs to compute loss, but it is set to None.')

    predictions = logits if loss_function_using_logits else probs
    if loss_function == LossFunction.CROSS_ENTROPY:
      loss = utils.log_loss(labels, predictions, loss_function_using_logits)
    elif loss_function == LossFunction.SQUARED:
      loss = utils.squared_loss(labels, predictions)
    else:
      loss = loss_function(labels, predictions)
    return loss

  def get_loss_train(self):
    """Calculates (if needed) cross-entropy losses for the training set.

    Returns:
      Loss (or None if neither the loss nor the labels are present).
    """
    if self.loss_function_using_logits is None:
      self.loss_function_using_logits = (self.logits_train is not None)
    return self._get_loss(self.loss_train, self.labels_train, self.logits_train,
                          self.probs_train, self.loss_function,
                          self.loss_function_using_logits)

  def get_loss_test(self):
    """Calculates (if needed) cross-entropy losses for the test set.

    Returns:
      Loss (or None if neither the loss nor the labels are present).
    """
    if self.loss_function_using_logits is None:
      self.loss_function_using_logits = bool(self.logits_test)
    return self._get_loss(self.loss_test, self.labels_test, self.logits_test,
                          self.probs_test, self.loss_function,
                          self.loss_function_using_logits)

  def get_entropy_train(self):
    """Calculates prediction entropy for the training set."""
    if self.entropy_train is not None:
      return self.entropy_train
    return self._get_entropy(self.logits_train, self.labels_train)

  def get_entropy_test(self):
    """Calculates prediction entropy for the test set."""
    if self.entropy_test is not None:
      return self.entropy_test
    return self._get_entropy(self.logits_test, self.labels_test)

  def get_train_size(self):
    """Returns size of the training set."""
    if self.loss_train is not None:
      return self.loss_train.size
    if self.entropy_train is not None:
      return self.entropy_train.size
    return self.logits_or_probs_train.shape[0]

  def get_test_size(self):
    """Returns size of the test set."""
    if self.loss_test is not None:
      return self.loss_test.size
    if self.entropy_test is not None:
      return self.entropy_test.size
    return self.logits_or_probs_test.shape[0]

  def validate(self):
    """Validates the inputs."""
    if (self.loss_train is None) != (self.loss_test is None):
      raise ValueError(
          'loss_test and loss_train should both be either set or unset')

    if (self.entropy_train is None) != (self.entropy_test is None):
      raise ValueError(
          'entropy_test and entropy_train should both be either set or unset')

    if (self.logits_train is None) != (self.logits_test is None):
      raise ValueError(
          'logits_train and logits_test should both be either set or unset')

    if (self.probs_train is None) != (self.probs_test is None):
      raise ValueError(
          'probs_train and probs_test should both be either set or unset')

    if (self.logits_train is not None) and (self.probs_train is not None):
      raise ValueError('Logits and probs can not be both set')

    if (self.labels_train is None) != (self.labels_test is None):
      raise ValueError(
          'labels_train and labels_test should both be either set or unset')

    if (self.labels_train is None and self.loss_train is None and
        self.logits_train is None and self.entropy_train is None and
        self.probs_train is None):
      raise ValueError(
          'At least one of labels, logits, losses, probabilities or entropy should be set'
      )

    if self.labels_train is not None and not _is_integer_type_array(
        self.labels_train):
      raise ValueError('labels_train elements should have integer type')

    if self.labels_test is not None and not _is_integer_type_array(
        self.labels_test):
      raise ValueError('labels_test elements should have integer type')

    _is_np_array(self.logits_train, 'logits_train')
    _is_np_array(self.logits_test, 'logits_test')
    _is_np_array(self.probs_train, 'probs_train')
    _is_np_array(self.probs_test, 'probs_test')
    _is_np_array(self.labels_train, 'labels_train')
    _is_np_array(self.labels_test, 'labels_test')
    _is_np_array(self.loss_train, 'loss_train')
    _is_np_array(self.loss_test, 'loss_test')
    _is_np_array(self.entropy_train, 'entropy_train')
    _is_np_array(self.entropy_test, 'entropy_test')

    _is_last_dim_equal(self.logits_train, 'logits_train', self.logits_test,
                       'logits_test')
    _is_last_dim_equal(self.probs_train, 'probs_train', self.probs_test,
                       'probs_test')
    _is_array_one_dimensional(self.loss_train, 'loss_train')
    _is_array_one_dimensional(self.loss_test, 'loss_test')
    _is_array_one_dimensional(self.entropy_train, 'entropy_train')
    _is_array_one_dimensional(self.entropy_test, 'entropy_test')
    _is_array_one_dimensional(self.labels_train, 'labels_train')
    _is_array_one_dimensional(self.labels_test, 'labels_test')

  def __str__(self):
    """Return the shapes of variables that are not None."""
    result = ['AttackInputData(']
    _append_array_shape(self.loss_train, 'loss_train', result)
    _append_array_shape(self.loss_test, 'loss_test', result)
    _append_array_shape(self.entropy_train, 'entropy_train', result)
    _append_array_shape(self.entropy_test, 'entropy_test', result)
    _append_array_shape(self.logits_train, 'logits_train', result)
    _append_array_shape(self.logits_test, 'logits_test', result)
    _append_array_shape(self.probs_train, 'probs_train', result)
    _append_array_shape(self.probs_test, 'probs_test', result)
    _append_array_shape(self.labels_train, 'labels_train', result)
    _append_array_shape(self.labels_test, 'labels_test', result)
    result.append(')')
    return '\n'.join(result)


def _append_array_shape(arr: Optional[np.ndarray], arr_name: str, result):
  if arr is not None:
    result.append(' %s with shape: %s,' % (arr_name, arr.shape))


@dataclasses.dataclass
class RocCurve:
  """Represents ROC curve of a membership inference classifier."""
  # Thresholds used to define points on ROC curve.
  # Thresholds are not explicitly part of the curve, and are stored for
  # debugging purposes.
  thresholds: np.ndarray

  # True positive rates based on thresholds
  tpr: np.ndarray

  # False positive rates based on thresholds
  fpr: np.ndarray

  def get_auc(self):
    """Calculates area under curve (aka AUC)."""
    return metrics.auc(self.fpr, self.tpr)

  def get_attacker_advantage(self):
    """Calculates membership attacker's (or adversary's) advantage.

    This metric is inspired by https://arxiv.org/abs/1709.01604, specifically
    by Definition 4. The difference here is that we calculate maximum advantage
    over all available classifier thresholds.

    Returns:
      a single float number with membership attacker's advantage.
    """
    return max(np.abs(self.tpr - self.fpr))

  def __str__(self):
    """Returns AUC and advantage metrics."""
    return '\n'.join([
        'RocCurve(',
        '  AUC: %.2f' % self.get_auc(),
        '  Attacker advantage: %.2f' % self.get_attacker_advantage(), ')'
    ])


# (no. of training examples, no. of test examples) for the test.
DataSize = collections.namedtuple('DataSize', 'ntrain ntest')


@dataclasses.dataclass
class SingleAttackResult:
  """Results from running a single attack."""

  # Data slice this result was calculated for.
  slice_spec: SingleSliceSpec

  # (no. of training examples, no. of test examples) for the test.
  data_size: DataSize
  attack_type: AttackType

  # NOTE: roc_curve could theoretically be derived from membership scores.
  # Currently, we store it explicitly since not all attack types support
  # membership scores.
  # TODO(b/175870479): Consider deriving ROC curve from the membership scores.

  # ROC curve representing the accuracy of the attacker
  roc_curve: RocCurve

  # Membership score is some measure of confidence of this attacker that
  # a particular sample is a member of the training set.
  #
  # This is NOT necessarily probability. The nature of this score depends on
  # the type of attacker. Scores from different attacker types are not directly
  # comparable, but can be compared in relative terms (e.g. considering order
  # imposed by this measure).
  #

  # Membership scores for the training set samples. For a perfect attacker,
  # all training samples will have higher scores than test samples.
  membership_scores_train: Optional[np.ndarray] = None

  # Membership scores for the test set samples. For a perfect attacker, all
  # test set samples will have lower scores than the training set samples.
  membership_scores_test: Optional[np.ndarray] = None

  def get_attacker_advantage(self):
    return self.roc_curve.get_attacker_advantage()

  def get_auc(self):
    return self.roc_curve.get_auc()

  def __str__(self):
    """Returns SliceSpec, AttackType, AUC and advantage metrics."""
    return '\n'.join([
        'SingleAttackResult(',
        '  SliceSpec: %s' % str(self.slice_spec),
        '  DataSize: (ntrain=%d, ntest=%d)' %
        (self.data_size.ntrain, self.data_size.ntest),
        '  AttackType: %s' % str(self.attack_type),
        '  AUC: %.2f' % self.get_auc(),
        '  Attacker advantage: %.2f' % self.get_attacker_advantage(), ')'
    ])


@dataclasses.dataclass
class SingleMembershipProbabilityResult:
  """Results from computing membership probabilities (denoted as privacy risk score in https://arxiv.org/abs/2003.10595).

  this part shows how to leverage membership probabilities to perform attacks
  with thresholding on them.
  """

  # Data slice this result was calculated for.
  slice_spec: SingleSliceSpec

  train_membership_probs: np.ndarray

  test_membership_probs: np.ndarray

  def attack_with_varied_thresholds(self, threshold_list):
    """Performs an attack with the specified thresholds.

    For each threshold value, we count how many training and test samples with
    membership probabilities larger than the threshold and further compute
    precision and recall values. We skip the threshold value if it is larger
    than every sample's membership probability.

    Args:
      threshold_list: List of provided thresholds

    Returns:
      An array of attack results.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        np.concatenate((np.ones(len(self.train_membership_probs)),
                        np.zeros(len(self.test_membership_probs)))),
        np.concatenate(
            (self.train_membership_probs, self.test_membership_probs)),
        drop_intermediate=False)

    precision_list = []
    recall_list = []
    meaningful_threshold_list = []
    max_prob = max(self.train_membership_probs.max(),
                   self.test_membership_probs.max())
    for threshold in threshold_list:
      if threshold <= max_prob:
        idx = np.argwhere(thresholds >= threshold)[-1][0]
        meaningful_threshold_list.append(threshold)
        precision_list.append(tpr[idx] / (tpr[idx] + fpr[idx]))
        recall_list.append(tpr[idx])

    return np.array(meaningful_threshold_list), np.array(
        precision_list), np.array(recall_list)

  def collect_results(self, threshold_list, return_roc_results=True):
    """The membership probability (from 0 to 1) represents each sample's probability of being in the training set.

    Usually, we choose a list of threshold values from 0.5 (uncertain of
    training or test) to 1 (100% certain of training)
    to compute corresponding attack precision and recall.

    Args:
      threshold_list: List of provided thresholds
      return_roc_results: Whether to return ROC results

    Returns:
      Summary string.
    """
    meaningful_threshold_list, precision_list, recall_list = self.attack_with_varied_thresholds(
        threshold_list)
    summary = []
    summary.append('\nMembership probability analysis over slice: \"%s\"' %
                   str(self.slice_spec))
    for i in range(len(meaningful_threshold_list)):
      summary.append(
          '  with %.4f as the threshold on membership probability, the precision-recall pair is (%.4f, %.4f)'
          % (meaningful_threshold_list[i], precision_list[i], recall_list[i]))
    if return_roc_results:
      fpr, tpr, thresholds = metrics.roc_curve(
          np.concatenate((np.ones(len(self.train_membership_probs)),
                          np.zeros(len(self.test_membership_probs)))),
          np.concatenate(
              (self.train_membership_probs, self.test_membership_probs)))
      roc_curve = RocCurve(tpr=tpr, fpr=fpr, thresholds=thresholds)
      summary.append(
          '  thresholding on membership probability achieved an AUC of %.2f' %
          (roc_curve.get_auc()))
      summary.append(
          '  thresholding on membership probability achieved an advantage of %.2f'
          % (roc_curve.get_attacker_advantage()))
    return summary


@dataclasses.dataclass
class MembershipProbabilityResults:
  """Membership probability results from multiple data slices."""

  membership_prob_results: Iterable[SingleMembershipProbabilityResult]

  def summary(self, threshold_list):
    """Returns the summary of membership probability analyses on all slices."""
    summary = []
    for single_result in self.membership_prob_results:
      single_summary = single_result.collect_results(threshold_list)
      summary.extend(single_summary)
    return '\n'.join(summary)


@dataclasses.dataclass
class PrivacyReportMetadata:
  """Metadata about the evaluated model.

  Used to create a privacy report based on AttackResults.
  """
  accuracy_train: Optional[float] = None
  accuracy_test: Optional[float] = None

  loss_train: Optional[float] = None
  loss_test: Optional[float] = None

  model_variant_label: str = 'Default model variant'
  epoch_num: Optional[int] = None


class AttackResultsDFColumns(enum.Enum):
  """Columns for the Pandas DataFrame that stores AttackResults metrics."""
  SLICE_FEATURE = 'slice feature'
  SLICE_VALUE = 'slice value'
  DATA_SIZE_TRAIN = 'train size'
  DATA_SIZE_TEST = 'test size'
  ATTACK_TYPE = 'attack type'

  def __str__(self):
    """Returns 'slice value' instead of AttackResultsDFColumns.SLICE_VALUE."""
    return '%s' % self.value


@dataclasses.dataclass
class AttackResults:
  """Results from running multiple attacks."""
  single_attack_results: MutableSequence[SingleAttackResult]

  privacy_report_metadata: Optional[PrivacyReportMetadata] = None

  def calculate_pd_dataframe(self):
    """Returns all metrics as a Pandas DataFrame."""
    slice_features = []
    slice_values = []
    data_size_train = []
    data_size_test = []
    attack_types = []
    advantages = []
    aucs = []

    for attack_result in self.single_attack_results:
      slice_spec = attack_result.slice_spec
      if slice_spec.entire_dataset:
        slice_feature, slice_value = str(slice_spec), ''
      else:
        slice_feature, slice_value = slice_spec.feature.value, slice_spec.value
      slice_features.append(str(slice_feature))
      slice_values.append(str(slice_value))
      data_size_train.append(attack_result.data_size.ntrain)
      data_size_test.append(attack_result.data_size.ntest)
      attack_types.append(str(attack_result.attack_type))
      advantages.append(float(attack_result.get_attacker_advantage()))
      aucs.append(float(attack_result.get_auc()))

    df = pd.DataFrame({
        str(AttackResultsDFColumns.SLICE_FEATURE): slice_features,
        str(AttackResultsDFColumns.SLICE_VALUE): slice_values,
        str(AttackResultsDFColumns.DATA_SIZE_TRAIN): data_size_train,
        str(AttackResultsDFColumns.DATA_SIZE_TEST): data_size_test,
        str(AttackResultsDFColumns.ATTACK_TYPE): attack_types,
        str(PrivacyMetric.ATTACKER_ADVANTAGE): advantages,
        str(PrivacyMetric.AUC): aucs
    })
    return df

  def summary(self, by_slices=False) -> str:
    """Provides a summary of the metrics.

    The summary provides the best-performing attacks for each requested data
    slice.
    Args:
      by_slices : whether to prepare a per-slice summary.

    Returns:
      A string with a summary of all the metrics.
    """
    summary = []

    # Summary over all slices
    max_auc_result_all = self.get_result_with_max_auc()
    summary.append('Best-performing attacks over all slices')
    summary.append(
        '  %s (with %d training and %d test examples) achieved an AUC of %.2f on slice %s'
        % (max_auc_result_all.attack_type, max_auc_result_all.data_size.ntrain,
           max_auc_result_all.data_size.ntest, max_auc_result_all.get_auc(),
           max_auc_result_all.slice_spec))

    max_advantage_result_all = self.get_result_with_max_attacker_advantage()
    summary.append(
        '  %s (with %d training and %d test examples) achieved an advantage of %.2f on slice %s'
        % (max_advantage_result_all.attack_type,
           max_advantage_result_all.data_size.ntrain,
           max_advantage_result_all.data_size.ntest,
           max_advantage_result_all.get_attacker_advantage(),
           max_advantage_result_all.slice_spec))

    slice_dict = self._group_results_by_slice()

    if by_slices and len(slice_dict.keys()) > 1:
      for slice_str in slice_dict:
        results = slice_dict[slice_str]
        summary.append('\nBest-performing attacks over slice: \"%s\"' %
                       slice_str)
        max_auc_result = results.get_result_with_max_auc()
        summary.append(
            '  %s (with %d training and %d test examples) achieved an AUC of %.2f'
            % (max_auc_result.attack_type, max_auc_result.data_size.ntrain,
               max_auc_result.data_size.ntest, max_auc_result.get_auc()))
        max_advantage_result = results.get_result_with_max_attacker_advantage()
        summary.append(
            '  %s (with %d training and %d test examples) achieved an advantage of %.2f'
            % (max_advantage_result.attack_type,
               max_advantage_result.data_size.ntrain,
               max_auc_result.data_size.ntest,
               max_advantage_result.get_attacker_advantage()))

    return '\n'.join(summary)

  def _group_results_by_slice(self):
    """Groups AttackResults into a dictionary keyed by the slice."""
    slice_dict = {}
    for attack_result in self.single_attack_results:
      slice_str = str(attack_result.slice_spec)
      if slice_str not in slice_dict:
        slice_dict[slice_str] = AttackResults([])
      slice_dict[slice_str].single_attack_results.append(attack_result)
    return slice_dict

  def get_result_with_max_auc(self) -> SingleAttackResult:
    """Get the result with maximum AUC for all attacks and slices."""
    aucs = [result.get_auc() for result in self.single_attack_results]

    if min(aucs) < 0.4:
      print('Suspiciously low AUC detected: %.2f. ' +
            'There might be a bug in the classifier' % min(aucs))

    return self.single_attack_results[np.argmax(aucs)]

  def get_result_with_max_attacker_advantage(self) -> SingleAttackResult:
    """Get the result with maximum advantage for all attacks and slices."""
    return self.single_attack_results[np.argmax([
        result.get_attacker_advantage() for result in self.single_attack_results
    ])]

  def save(self, filepath):
    """Saves self to a pickle file."""
    with open(filepath, 'wb') as out:
      pickle.dump(self, out)

  @classmethod
  def load(cls, filepath):
    """Loads AttackResults from a pickle file."""
    with open(filepath, 'rb') as inp:
      return pickle.load(inp)


@dataclasses.dataclass
class AttackResultsCollection:
  """A collection of AttackResults."""
  attack_results_list: MutableSequence[AttackResults]

  def append(self, attack_results: AttackResults):
    self.attack_results_list.append(attack_results)

  def save(self, dirname):
    """Saves self to a pickle file."""
    for i, attack_results in enumerate(self.attack_results_list):
      filepath = os.path.join(dirname,
                              _get_attack_results_filename(attack_results, i))

      attack_results.save(filepath)

  @classmethod
  def load(cls, dirname):
    """Loads AttackResultsCollection from all files in a directory."""
    loaded_collection = AttackResultsCollection([])
    for filepath in sorted(glob.glob('%s/*' % dirname)):
      with open(filepath, 'rb') as inp:
        loaded_collection.attack_results_list.append(pickle.load(inp))
    return loaded_collection


def _get_attack_results_filename(attack_results: AttackResults, index: int):
  """Creates a filename for a specific set of AttackResults."""
  metadata = attack_results.privacy_report_metadata
  if metadata is not None:
    return '%s_%s_epoch_%s.pickle' % (metadata.model_variant_label, index,
                                      metadata.epoch_num)
  return '%s.pickle' % index


def get_flattened_attack_metrics(results: AttackResults):
  """Get flattened attack metrics.

  Args:
    results: membership inference attack results.

  Returns:
       types: a list of attack types
       slices: a list of slices
       attack_metrics: a list of metric names
       values: a list of metric values, i-th element correspond to properties[i]
  """
  types = []
  slices = []
  attack_metrics = []
  values = []
  for attack_result in results.single_attack_results:
    types += [str(attack_result.attack_type)] * 2
    slices += [str(attack_result.slice_spec)] * 2
    attack_metrics += ['adv', 'auc']
    values += [
        float(attack_result.get_attacker_advantage()),
        float(attack_result.get_auc())
    ]
  return types, slices, attack_metrics, values

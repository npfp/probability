# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for Frechet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class _FrechetTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self._dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self._use_static_shape else None)

  def testFrechetShape(self):
    loc = np.array([3.0] * 5, dtype=self._dtype)
    scale = np.array([3.0] * 5, dtype=self._dtype)
    concentration = np.array([3.0] * 5, dtype=self._dtype)
    frechet = tfd.Frechet(loc=loc, scale=scale, concentration=concentration,
                          validate_args=True)

    self.assertEqual((5,), self.evaluate(frechet.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), frechet.batch_shape)
    self.assertAllEqual([], self.evaluate(frechet.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), frechet.event_shape)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      frechet = tfd.Frechet(loc=0., scale=scale, concentration=1.,
                           validate_args=True)
      self.evaluate(frechet.mean())

    scale = tf.Variable([.01])
    self.evaluate(scale.initializer)
    frechet = tfd.Frechet(loc=0., scale=scale, concentration=1.0,
                          validate_args=True)
    self.assertIs(scale, frechet.scale)
    self.evaluate(frechet.mean())
    with tf.control_dependencies([scale.assign([-.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(frechet.mean())

  def testInvalidConcentration(self):
    concentration = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      frechet = tfd.Frechet(loc=0., scale=1.0, concentration=concentration,
                           validate_args=True)
      self.evaluate(frechet.mean())

    concentration = tf.Variable([.01])
    self.evaluate(concentration.initializer)
    frechet = tfd.Frechet(loc=0., scale=1.0, concentration=concentration,
                          validate_args=True)
    self.assertIs(concentration, frechet.concentration)
    self.evaluate(frechet.mean())
    with tf.control_dependencies([concentration.assign([-.01])]):
      with self.assertRaisesOpError('Argument `concentration` must be positive.'):
        self.evaluate(frechet.mean())

  def testFrechetLogPdf(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    concentration = np.array([1.5] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)
    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)
    log_pdf = frechet.log_prob(self.make_tensor(x))
    self.assertAllClose(
        stats.invweibull.logpdf(
          x, loc=loc, scale=scale, c=concentration),
        self.evaluate(log_pdf))

    pdf = frechet.prob(x)
    self.assertAllClose(
        stats.invweibull.pdf(
          x, loc=loc, scale=scale, c=concentration)
      , self.evaluate(pdf))

  def testFrechetLogPdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[-1.0, 0.0, 1.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    concentration = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)
    log_pdf = frechet.log_prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_pdf),
      stats.invweibull.logpdfa(x, loc=loc, scale=scale, c=concentration))

    pdf = frechet.prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(pdf),
      stats.gumbel_r.pdf(x, loc=loc, scale=scale, c=concentration))

  def testFrechetCDF(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    concentration = np.array([2.] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    log_cdf = frechet.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
      stats.invweibull.logcdf(x, loc=loc, scale=scale, c=concentration))

    cdf = frechet.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
      stats.invweibull.cdf(x, loc=loc, scale=scale, c=concentration))

  def testFrechetCdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[-1.0, 1.0, 1.5]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    concentration = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    log_cdf = frechet.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.invweibull.logcdf(x, loc=loc, scale=scale, c=concentration))

    cdf = frechet.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.invweibull.cdf(x, loc=loc, scale=scale, c=concentration))

  def testFrechetMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    #TODO(npfp):  test concentration <= 1.
    concentration = np.array([1.5], dtype=self._dtype)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)
    self.assertAllClose(
      self.evaluate(frechet.mean()),
      stats.invweibull.mean(loc=loc, scale=scale, c=concentration))

  def testFrechetVariance(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    #TODO(npfp):  test concentration <= 2.
    concentration = np.array([2.5], dtype=self._dtype)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    self.assertAllClose(
      self.evaluate(frechet.variance()),
      stats.invweibull.var(loc=loc, scale=scale, c=concentration))

  def testFrechetStd(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    #TODO(npfp):  test concentration <= 2.
    concentration = np.array([2.5], dtype=self._dtype)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    self.assertAllClose(
      self.evaluate(frechet.stddev()),
      stats.invweibull.std(loc=loc, scale=scale, c=concentration))

  def testFrechetMode(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    concentration = np.array([2.5], dtype=self._dtype)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    self.assertAllClose(
      self.evaluate(frechet.mode()),
      stats.invweibull.mode(loc=loc, scale=scale, c=concentration))

  def testFrechetSample(self):
    loc = self._dtype(1.0)
    scale = self._dtype(1.0)
    concentration = self._dtype(2.25)
    n = int(100e3)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    samples = frechet.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), sample_values.shape)
    self.assertAllClose(
        stats.invweibull.mean(loc=loc, scale=scale, c=concentration),
        sample_values.mean(), rtol=.01)
    self.assertAllClose(
        stats.invweibull.var(loc=loc, scale=scale, c=concentration),
        sample_values.var(), rtol=.01)

  def testFrechetSampleMultidimensionalMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    concentration = np.array([3.5, 2.5, 1.5], dtype=self._dtype)
    n = int(1e5)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    samples = frechet.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.invweibull.mean(loc=loc, scale=scale, c=concentration),
        sample_values.mean(axis=0),
        rtol=.03,
        atol=0)

  def testFrechetSampleMultidimensionalVar(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    concentration = np.array([4.5, 2.5, 3.5], dtype=self._dtype)
    n = int(1e5)

    frechet = tfd.Frechet(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(concentration),
        validate_args=True)

    samples = frechet.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.invweibull.var(loc=loc, scale=scale, c=concentration),
        sample_values.var(axis=0),
        rtol=.03,
        atol=0)

  def testFrechetSampleMultidimensionalConcentration(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    concentration = np.array([[3.5, 2.5, 1.5]], dtype=self._dtype)
    n = int(1e5)

    frechet = tfd.Frechet(
      loc=self.make_tensor(loc),
      scale=self.make_tensor(scale),
      concentration=self.make_tensor(concentration),
      validate_args=True)

    samples = frechet.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
      stats.invweibull.mean(loc=loc, scale=scale, c=concentration),
      sample_values.mean(axis=0),
      rtol=.03,
      atol=0)

@test_util.test_all_tf_execution_regimes
class FrechetTestStaticShape(test_util.TestCase, _FrechetTest):
  _dtype = np.float32
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class FrechetTestFloat64StaticShape(test_util.TestCase, _FrechetTest):
  _dtype = np.float64
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class FrechetTestDynamicShape(test_util.TestCase, _FrechetTest):
  _dtype = np.float32
  _use_static_shape = False


if __name__ == '__main__':
  tf.test.main()

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
"""The Frechet distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import frechet_cdf as frechet_cdf_bijector
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util, \
  prefer_static
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


def gamma(x, name=None):
  name = name or "Gamma"
  with tf.name_scope(name):
    return tf.math.exp(tf.math.lgamma(x))


class Frechet(transformed_distribution.TransformedDistribution):
  """The scalar Frechet distribution.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma, alpha) =
  ```

  where `loc = mu`, `scale = sigma` and `alpha = concentration`.

  The cumulative density function of this distribution is,

  ```cdf(x; mu, sigma, alpha) = exp(-((x - mu) / sigma)) ** alpha)```

  The Frechet distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Frechet(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Frechet distribution.
  dist = tfd.Frechet(loc=0., scale=3., concentration=1.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Frechets.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tfd.Frechet(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Logistics.
  # Both have mean 1, but different scales.
  dist = tfd.Frechet(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               scale,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='Frechet'):
    """Construct Frechet distributions with location and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s).
        scale must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Frechet'`.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, concentration],
                                      dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      concentration = tensor_util.convert_nonref_to_tensor(
        concentration, name='concentration', dtype=dtype)
      dtype_util.assert_same_float_dtype([loc, scale, concentration])
      # Positive scale and concentration are asserted
      # by the incorporated Frechet bijector.
      self._frechet_bijector = frechet_cdf_bijector.FrechetCDF(
          loc=loc, scale=scale, concentration=concentration,
          validate_args=validate_args)

      # Because the uniform sampler generates samples in `[0, 1)` this would
      # cause samples to lie in `(inf, -inf]` instead of `(inf, -inf)`. To fix
      # this, we use `np.finfo(dtype_util.as_numpy_dtype(self.dtype).tiny`
      # because it is the smallest, positive, 'normal' number.
      batch_shape = distribution_util.get_broadcast_shape(
        loc, scale, concentration)
      super(Frechet, self).__init__(
          # TODO(b/137665504): Use batch-adding meta-distribution to set the
          # batch shape instead of tf.ones.
          distribution=uniform.Uniform(
              low=np.finfo(dtype_util.as_numpy_dtype(dtype)).tiny,
              high=tf.ones(batch_shape, dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          # The Frechet bijector encodes the CDF function as the forward,
          # and hence needs to be inverted.
          bijector=invert_bijector.Invert(
              self._frechet_bijector, validate_args=validate_args),
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'concentration'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0, concentration=0)

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._frechet_bijector.loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._frechet_bijector.scale

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._frechet_bijector.concentration

  def _z(self, x):
    return self._frechet_bijector._z(x)  # pylint: disable=protected-access

  def _log_cdf(self, x):
    z = tf.convert_to_tensor(self._z(x), dtype=self.dtype)
    return - tf.math.exp(- self.concentration * tf.math.log(z))

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return (1. + np.euler_gamma + np.euler_gamma / self.concentration
            + tf.math.log(scale/self.concentration))

  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    return tf.where(
      concentration > 1.,
      self.loc + self.scale * gamma(1 - 1./self.concentration),
      np.infty)

  def _variance(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    concentration = tf.convert_to_tensor(self.concentration)
    scale = self.scale * tf.ones_like(self.loc)
    return tf.where(
      concentration > 2.,
      (gamma(1. - 2./self.concentration)
       - gamma(1. - 1./self.concentration) ** 2
       ) * scale ** 2,
      np.infty)

  def _mode(self):
    return (self.loc + self.scale * (
        self.concentration / (1. + self.concentration)
    ) ** (1./self.concentration))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return self._frechet_bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access

import numpy as np
import pandas as pd
# tf.debugging.set_log_device_placement(True)


from abc import ABC, abstractmethod
import datetime
from itertools import chain
import time
import copy
import os
from typing import Dict, List, Tuple, Union

import pesuelogit
from nesuelogit.metrics import error, mse, sse, rmse, nrmse, nmse, btcg_mse, mnrmse, l1norm, mape, r2_score, zscore, z2score
from nesuelogit.utils import timeit
from pesuelogit.models import PESUELOGIT, normalize_od, compute_rr

import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

# Regression Krigging
from sklearn.linear_model import LinearRegression
from pykrige.rk import RegressionKriging
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

import statsmodels.api as sm

from isuelogit.printer import block_output, printIterationBar
import isuelogit as isl
import copy


# tf.config.set_soft_device_placement(True)

def mask_observed_flow(flow, D):
    # Remove cases where there are no path traversing the link
    observed_flow = flow.numpy()
    for i in range(observed_flow.shape[0]):
        observed_flow[i, :] = observed_flow[i, :] * tf.cast(tf.reduce_sum(D, 1) >= 1, flow.dtype)

    return tf.constant(observed_flow)


def mask_observed_traveltime(traveltime, k, tt_ff, k_threshold=1e5):

    mask1 = np.where((k >= k_threshold), float('nan'), 1)
    mask2 = np.where((tt_ff == 0), float('nan'), 1)

    return mask2 * mask1 * traveltime

def mask_Y(Y, k, tt_ff, D):
    observed_traveltime, observed_flow = tf.unstack(Y, axis=-1)
    observed_traveltime = mask_observed_traveltime(observed_traveltime, k = k, tt_ff = tt_ff)
    observed_flow =mask_observed_flow(observed_flow, D = D)

    return tf.stack([observed_traveltime, observed_flow], axis=2)

def get_period_ids(period_column: np.array, period_dict=None):
    """

    :param period_column: 3D array, with dimensions (n_timepoints,n_links,1)
    :param period_dict:
    :return:
    """

    period_ids = period_column.copy()

    if period_dict is None:
        period_keys = np.unique(period_ids)

        # This dictionary can map the original period ids into the indices of the array used internally by the model
        period_dict = dict(zip(list(map(int, np.sort(period_keys))), range(0, len(period_keys))))

    for i, j in period_dict.items():
        period_ids[period_ids == i] = j

    return period_ids, period_dict


def compute_generated_trips(q, ods, **kwargs):
    od_parameters = ODParameters(ods=ods, trainable=False, initial_values=None)

    return od_parameters.compute_generated_trips(values=q, **kwargs)


def compute_generation_factors(period_column: np.array, flow_column: np.array, reference_period: int, period_dict=None):
    period_ids, period_dict = get_period_ids(period_column=period_column, period_dict=period_dict)

    n_timepoints, n_links, _ = period_ids.shape

    df = pd.DataFrame(tf.reshape(tf.concat([period_ids, flow_column], axis=-1), [-1, 2]))
    df.columns = ['period_id', 'flow']

    generation_factors = df.groupby('period_id')['flow'].mean() / \
                         df[df.period_id == period_dict[reference_period]].flow.mean()

    return generation_factors


def normalized_losses(losses: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in losses.columns if col != 'epoch']

    losses[columns] = losses[columns] / losses[losses['epoch'] == losses['epoch'].min()][columns].values

    return losses

def utility_parameters_periods(model, period_feature, period_keys, include_vot=False):
    theta_df = pd.DataFrame({})

    period_dict = {v: k for k, v in model.period_dict.items()}

    for i in range(model.theta.shape[0]):
        theta_dict = dict(zip(model.utility.features, list(model.theta[i].numpy())))

        if include_vot:
            theta_dict['vot'] = float(compute_rr(theta_dict))

        label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
        label_period_feature_2 = label_period_feature_1 + 1

        label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"

        theta_dict[period_feature] = label_period_feature_1

        theta_df = pd.concat([theta_df, pd.DataFrame(theta_dict, index=[label_period_feature])])

    if include_vot:
        theta_df[theta_df['vot'].isna()] = 0

    theta_df = theta_df.sort_values(period_feature)

    cols = theta_df.columns
    theta_df[cols] = theta_df[cols].apply(pd.to_numeric, errors='coerce')

    return theta_df


class Parameters(pesuelogit.models.Parameters):

    def __init__(self, dtype, *args, **kwargs):

        # Cast to right daya type
        self.dtype = dtype

        if 'initial_values' in kwargs.keys():
            for k, v in kwargs['initial_values'].items():
                if isinstance(v, np.ndarray):
                    kwargs['initial_values'][k] = tf.convert_to_tensor(kwargs['initial_values'][k], self.dtype)

                kwargs['initial_values'][k] = tf.cast(kwargs['initial_values'][k], self.dtype)

        if 'true_values' in kwargs.keys():
            for k, v in kwargs['true_values'].items():
                if isinstance(v, np.ndarray):
                    kwargs['true_values'][k] = tf.convert_to_tensor(kwargs['true_values'][k], self.dtype)

                kwargs['true_values'][k] = tf.cast(kwargs['true_values'][k], self.dtype)

        super().__init__(*args, **kwargs)


class UtilityParameters(Parameters):

    def __init__(self, time_varying=False, n_classes=1, n_periods=1, *args, **kwargs):
        kwargs['keys'] = kwargs.get('keys', [])
        kwargs['keys'] += ['fixed_effect', 'psc_factor']

        super().__init__(*args, **kwargs, n_periods=n_periods)

        self.n_classes = n_classes
        self.time_varying = time_varying

    def initial_values_array(self, features=None) -> np.array:
        # TODO: Add a new dimension to account for multiple classes of users.

        values_list = list(self.initial_values.values())

        if features is not None:
            values_list = [self.initial_values[feature] for feature in features]

        # if self.n_periods == 1:
        #     return np.array(list(values_list))

        return np.repeat(np.array(values_list)[np.newaxis, :], self.n_periods, axis=0)


class GenerationParameters(UtilityParameters):

    def __init__(self,
                 pretrain_generation_weights=False,
                 historic_g=None,
                 *args,
                 **kwargs):
        super().__init__(features_Y=None, *args, **kwargs)

        self.historic_g = historic_g

        if self.historic_g is not None:
            self.historic_g = tf.cast(self.historic_g, self.dtype)

        self._pretrain_generation_weights = pretrain_generation_weights

    @property
    def pretrain_generation_weights(self):
        return self._pretrain_generation_weights


class ODParameters(pesuelogit.models.ODParameters):
    """ Support OD with multiple periods """

    def __init__(self,
                 ods: List[Tuple],
                 features_generation: List[str] = None,
                 features_distribution: List[str] = None,
                 n_periods=1,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs, n_periods=n_periods)

        self.ods = ods

        # Sparse tensor generated from OD matrix of size |N| x |N|. Cells take value 1 if there are trips, 0, otherwise.
        self._L_sparse = tf.sparse.SparseTensor(indices=self.ods,
                                                values=tf.ones_like(range(0, len(self.ods))),
                                                dense_shape=(len(self.nodes), len(self.nodes)))

        self._features_generation = features_generation
        self._features_distribution = features_distribution

    @property
    def  historic_values_array(self):
        # historic_od = tf.expand_dims(tf.constant(self.network.q.flatten()), axis=0)
        # if len(list(self.historic_values.keys())) > 1:

        if isinstance(self.historic_values, dict) and self.historic_values is not None:
            # historic_od = np.empty((self.n_periods, list(self.historic_values.values())[0].shape[0]))
            historic_od = np.empty(self.initial_value.shape)
            historic_od[:] = np.nan

            for period, od in self.historic_values.items():
                historic_od[period, :] = od

            return historic_od
        # return self._historic_values

        else:
            return self.historic_values

    @property
    def features_generation(self):
        return self._features_generation

    @features_generation.setter
    def features_generation(self, value):
        self._features_generation = value

    @property
    def features_distribution(self):
        return self._features_distribution

    @features_distribution.setter
    def features_distribution(self, value):
        self._features_distribution = value

    @property
    def nodes(self):
        cur_min = 0
        cur_max = 0

        for i, j in self.ods:
            if max(i, j) >= cur_max:
                cur_max = max(i, j)
            if min(i, j) <= cur_min:
                cur_min = min(i, j)

        return np.arange(cur_min, cur_max + 1)

    @property
    def L_sparse(self):
        return self._L_sparse

    def get_sparse_tensor(self, values=None):

        if values is None:
            values = self.initial_values_array()

        # return tf.sparse.SparseTensor(indices=[self.ods] * values.shape[0],
        #                               values=values.flatten(),
        #                               dense_shape=(values.shape[0],len(self.nodes), len(self.nodes)))

        # tf.sparse.concat(0, [tf.sparse.expand_dims(M_sparse, 0)] * vf.shape[0])

        periods = values.shape[0]

        indices = tf.concat([
            tf.expand_dims(tf.repeat(np.array(range(0, periods)), (len(self.ods))), 1),
            list(chain(*[self.ods] * periods))
        ], 1)

        return tf.sparse.SparseTensor(indices=indices,
                                      values=tf.reshape(values, -1),
                                      dense_shape=(periods, len(self.nodes), len(self.nodes)))

    def compute_generated_trips(self, values=None, **kwargs):

        # tf.sparse.to_dense(self.get_sparse_tensor(values))
        #
        # a = self.get_sparse_tensor(values)

        # tf.reduce_sum(np.array([[1,2],[3,4]]),axis = 0)

        return tf.sparse.reduce_sum(self.get_sparse_tensor(values), axis=2, **kwargs)

    def compute_attracted_trips(self, values=None, **kwargs):

        return tf.sparse.reduce_sum(self.get_sparse_tensor(values), axis=1, **kwargs)

    def compute_trips_proportions(self, values=None):
        '''
        This method is used to initialize the fixed effect of the trip distribution equation

        :param historic_od:
        :return:
        '''

        if values is None:
            values = self.initial_values_array().flatten()

        return values / self.compute_generated_trips(values, keepdims=True)


class PerformanceFunction(tf.keras.layers.Layer):

    def __init__(self, type, max_traveltime_factor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.type = type
        self.max_traveltime_factor = max_traveltime_factor

    # @abstractmethod
    # def call(self, **kwargs):
    #     pass

    def build(self):
        raise NotImplementedError


class BiasConstraint(tf.keras.constraints.Constraint):

    def __init__(self, free_flow_traveltimes):
        self.free_flow_traveltimes = free_flow_traveltimes

    def __call__(self, b):
        # return tf.linalg.diag_part(w)*tf.eye(w.shape[0])
        return tf.clip_by_value(b, clip_value_min=self.free_flow_traveltimes,
                                clip_value_max=self.free_flow_traveltimes * 10)


class KernelConstraint(tf.keras.constraints.Constraint):

    def __init__(self,
                 capacities,
                 free_flow_traveltimes,
                 diagonal: bool = False,
                 symmetric: bool = False,
                 homogenous: bool = False,
                 initial_values = None,
                 bounds_clipping: [float,float] = None,
                 min_diagonal_value = 1e-8,
                 link_keys: List[Tuple[int, int]] = None,
                 dtype=tf.float32,
                 adjacency_constraint: bool = False):

        self.capacities = capacities
        self.free_flow_traveltimes = free_flow_traveltimes
        self.diagonal = diagonal
        self.symmetric = symmetric
        self.homogenous = homogenous
        self.n_links = len(capacities)
        self.link_keys = link_keys
        self.dtype = dtype
        self.constraint_matrix = None

        if bounds_clipping is None:
            bounds_clipping = [-10, 10]

        # Clip all values of the matrix between bounds
        self.bounds_clipping = bounds_clipping

        assert self.bounds_clipping[0] < self.bounds_clipping[1], \
            "first element in clipping must be lower than second element"

        self.min_diagonal_value = min_diagonal_value

        assert self.min_diagonal_value>=self.bounds_clipping[0] and  self.min_diagonal_value <=self.bounds_clipping[1], \
            "Minimum value of diagonal must be between the bounds defined by bounds_clipping argument"

        self.adjacency_constraint = adjacency_constraint

        if self.adjacency_constraint:
            self.adjacency_constraint_matrix \
                = self.create_adjacency_constraint_matrix(link_keys=self.link_keys, dtype=self.dtype)

        self.initial_values = initial_values

        if self.initial_values is None:
            # kernel_initializer = tf.constant_initializer(0)

            # self.kernel_initializer = tf.constant_initializer(0.15 * np.eye(self.n_links))

            # Initialize to one works the best
            # self.kernel_initializer = tf.constant_initializer(1 * np.eye(n_links))

            # self.kernel_constraint.initial_values = 1e-1*tf.ones((self.n_links,self.n_links), dtype = tf.float32)
            # self.kernel_constraint.initial_values = tf.ones((self.n_links, self.n_links), dtype=tf.float32)
            # Assign 0 as initial values of the adjacency constraint terms
            self.initial_values = tf.ones((self.n_links,self.n_links), dtype=self.dtype)

        self.constraint_matrix = tf.ones((self.n_links, self.n_links), dtype=self.dtype)

        if self.adjacency_constraint:
            self.constraint_matrix *= self.adjacency_constraint_matrix

        if self.diagonal or self.homogenous:
            self.constraint_matrix *= tf.eye(self.n_links)

        self.initial_values *= self.constraint_matrix


    def create_adjacency_constraint_matrix(self, link_keys: List[Tuple], dtype=None):

        if dtype is None:
            self.dtype = dtype

        n_links = len(link_keys)

        # self.adjacency_constraint = tf.eye(self.n_links, dtype = dtype)
        matrix = tf.zeros((n_links, n_links), dtype=dtype)

        heads, tails = np.array(link_keys).T
        counter = 0
        for head, tail in link_keys:
            idxs = np.where((heads == head) | (tails == head) | (heads == tail) | (tails == tail))[0]
            idxs = [(counter, idx) for idx in idxs]

            matrix = tf.tensor_scatter_nd_update(matrix, idxs, tf.ones(len(idxs)))
            counter += 1

            # tf.reduce_sum(self.adjacency_constraint, axis=1)

        return matrix


    def __call__(self, W):
        # return tf.linalg.diag_part(w)*tf.eye(w.shape[0])
        # return tf.clip_by_value(tf.linalg.diag_part(w) * tf.eye(w.shape[0]), clip_value_min=0, clip_value_max=1)

        # Minimum value in the diagonal. Assumption: higher link flow, higher travel time -> kernel diagonal has non-negative elements
        self.min_diagonal_element = self.min_diagonal_value

        if self.symmetric:
            W_upper = tf.compat.v1.matrix_band_part(W, 0, -1)
            factor_interaction = 1  # with -1 Symmetric but with opposite signs
            W = 0.5 * (W_upper + factor_interaction * tf.transpose(W_upper))
            # Note that the operation below will not affect the symmetry of the matrix

        W = tf.clip_by_value(W,
                         clip_value_min=self.bounds_clipping[0],
                         # clip_value_min=0,
                         clip_value_max=self.bounds_clipping[1]
                         # clip_value_max=10
                         )

        W_old_diagonal = tf.linalg.diag_part(W) * tf.eye(W.shape[0])

        # W_new_diagonal = tf.linalg.diag_part(W)* tf.eye(W.shape[0])
        W_new_diagonal = tf.linalg.diag_part(tf.clip_by_value(W_old_diagonal,
                                          clip_value_min= self.min_diagonal_element,
                                          # clip_value_min=0,
                                          clip_value_max= self.bounds_clipping[1],
                                          # clip_value_max=10

                                          )) * tf.eye(W.shape[0])
        # if self.homogenous:
        #     # Only average over terms that are greater than zero.
        #     self.W = tf.cast(tf.where(self.W > 0, 1, 0), self.W.dtype) * tf.reduce_mean(self.W[self.W > 0])

        if self.diagonal:
            self.W = W_new_diagonal

            return self.W
        else:
            W = (W - W_old_diagonal + W_new_diagonal)

        if self.adjacency_constraint:
            W *= self.adjacency_constraint_matrix

        self.W = W

        return self.W

class PolynomialLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dtype = tf.float32,
                 poly_order = 1,
                 trainable = True,
                 pretrain_weights = False,
                 alpha_prior = 0.15,
                 beta_prior = 4,
                 kernel_constraint = None,
                 link_specific = False,
                 alpha_relu = 0,
                 *args,
                 **kwargs
                 ):

        kwargs['dtype'] = dtype
        super().__init__(*args, **kwargs)

        self.poly_order = poly_order
        self.poly = PolynomialFeatures(self.poly_order, include_bias=False)
        self._built = False
        self.trainable_layer = trainable
        self._pretrain_weights = pretrain_weights
        self._pretrained_weights = False
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.kernel_constraint = kernel_constraint
        self.alpha_relu = alpha_relu
        self.link_specific = link_specific

    @property
    def poly_weights(self):
        return tf.exp(self._poly_weights)

    def build(self, n_poly_features = None, n_links = None):

        if n_links is None:
            n_links = self.n_links

        if n_poly_features is None:
            n_poly_features = self.poly_order

        rows = 1
        if self.link_specific:
            rows = n_links

        # Normalize by the number of poly features to avoid the initial polynomial to blow up
        # initial_value = np.log((1/n_poly_features)*np.ones((rows, n_poly_features), dtype = self.dtype))
        initial_value = np.log(np.ones((rows, n_poly_features), dtype=self.dtype))

        self._poly_weights = tf.Variable(
            # self.bpr.parameters['alpha'].initial_value,
            # initial_value=np.log(self.initial_values['alpha']),
            initial_value= initial_value,
            # np.sqrt(self.bpr.parameters['alpha'].initial_value),
            trainable= self.trainable_layer,
            name='poly_weights',
            dtype=self.dtype)

        # units = 1
        # if self.link_specific:
        #     units = n_links
        #
        # self.model = tf.keras.models.Sequential([
        #     tf.keras.Input(shape=(n_links, n_poly_features))])
        #
        # self.model.add(tf.keras.layers.Dense(n_links,
        #                                      name ='polynomial_kernel',
        #                                      use_bias=False,
        #                                      kernel_constraint = self.kernel_constraint
        #                                      # bias_constraint = tf.keras.constraints.NonNeg(),
        #                                      # kernel_initializer = tf.keras.initializers.Constant(1e-4)
        #                                      # kernel_initializer=tf.keras.initializers.Constant(1/n_poly_features)
        #                                      # kernel_initializer=tf.keras.initializers.Constant(0)
        #                                      )
        #                )
        #
        # # Note that ReLU layer does not have any effect if the kernel constraint restricts the weights to be positive
        # # self.model.add(tf.keras.layers.ReLU())
        # self.model.add(tf.keras.layers.LeakyReLU(self.alpha_relu))
        # # self.model.add(tf.keras.layers.PReLU())

        self._built = True
        # self.model.trainable = self.trainable_layer

    def map_flows(self, flows, capacities):

        """
        :param flows:
        :return:
        """

        # self.poly = PolynomialFeatures(2)
        # if self.poly_order > 1:

        flows /= capacities

        flows = tf.cast(tf.concat([list(map(lambda x: self.poly.fit_transform(tf.transpose(x)),
                                            tf.split(flows, flows.shape[0])))], axis =2), dtype = self.dtype)

        return flows

    def transform_flows(self, flows, capacities, period_ids, n_periods = None):

        flows = self.map_flows(flows = flows, capacities=capacities)
        n_links = len(capacities)

        if not self._built:
            self.n_poly_features = flows.shape[-1]
            self.n_periods = flows.shape[0]
            self.build()

        if n_periods is None:
            n_periods = self.n_periods
        #
        # if self.link_specific:
        #     return tf.reduce_sum(flows,2)

        # flows = tf.reshape(self.model(flows), (n_periods, n_links))
        if self.link_specific:
            flows = tf.einsum("ijk,jk -> ij", flows, self.poly_weights)
        else:
            flows = tf.einsum("ijk,lk -> ij", flows, self.poly_weights)

        return tf.cast(tf.experimental.numpy.take(flows, tf.cast(period_ids[:,0], dtype=tf.int32), 0), self.dtype)

    def pretrain_weights(self, flows, free_flow_traveltimes, capacities):
        """

        Weights are pretrained by mimicking a BPR with parameters alpha, beta

        :param flows:
        :param free_flow_traveltimes:
        :param capacities:
        :return:
        """
        # TODO: Alternatively, we can use observed flows matched with travel times.

        tt_ff = free_flow_traveltimes

        t = tt_ff * (1 + self.alpha_prior * tf.math.pow(flows / capacities, self.beta_prior))

        y = (t/tt_ff-1)/self.alpha_prior
        X = self.map_flows(flows, capacities = capacities)

        # Reshape array to match sklearn format
        y = y.numpy().reshape(-1,1)
        X = X.numpy().reshape(-1,X.shape[2])

        # Remove rows with nan values
        X,y = X[(~np.isnan(y)).flatten()], y[~np.isnan(y).flatten()]

        positive = False
        if isinstance(self.kernel_constraint, tf.keras.constraints.NonNeg):
            positive = True

        reg = LinearRegression(fit_intercept=False, positive = positive).fit(X = X, y = y)
        # reg = LinearRegression(fit_intercept=False).fit(y, X)

        coefs = reg.coef_

        if not self._built:
            self.build(n_poly_features=X.shape[-1], n_links=X.shape[1])

        # To ensure monotonicity
        # self.weights[0].assign(tf.nn.relu(coefs.reshape(self.weights[0].shape)))

        # self.weights[0].assign(coefs.reshape(self.weights[0].shape))
        with block_output(show_stdout=False, show_stderr=False):
            self._poly_weights.assign(np.log(tf.repeat(coefs, self._poly_weights.shape[0], axis =0)))

        # self.model.trainable = self.trainable_layer

        # self._pretrain_weights = False

        # poly_weights = np.array2string(self.weights[0].numpy().flatten(),
        #                                formatter={'float': lambda x: f'{x:.1e}'})

        poly_weights = np.array2string(
            np.mean(self.poly_weights.numpy(), axis=0),
            formatter={'float': lambda x: f'{x:.1e}'})

        print(f"\nPretrained polynomial weights: {poly_weights}\n", end='')


class SingleWeightLayer(tf.keras.layers.Layer):
    def __init__(self,
                 initial_value = 1,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.initial_value = initial_value

    def build(self, input_shape):
        self._parameter = tf.Variable(np.log(self.initial_value),
                                      dtype=self.dtype,
                                      # constraint = tf.keras.constraints.NonNeg()
                                      )
    @property
    def parameter(self):
        return tf.exp(self._parameter)

    def call(self, input):
        return self.parameter*input

class KernelWeightLayer(tf.keras.layers.Layer):
    def __init__(self,
                 initial_value_diagonal = None,
                 initial_value_nondiagonal = None,
                 n_links = None,
                 kernel_constraint = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.initial_value_diagonal = initial_value_diagonal
        self.initial_value_nondiagonal = initial_value_nondiagonal
        self.n_links = n_links
        self._kernel_matrix = None
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        self._diagonal_parameters = tf.Variable(np.log(self.initial_value_diagonal*tf.ones(self.n_links)),
                                      dtype=self.dtype,
                                      name = 'diagonal_kernel_matrix'
                                      # constraint = tf.keras.constraints.NonNeg()
                                      )

        self._nondiagonal_parameters = tf.Variable(self.initial_value_nondiagonal*tf.ones((self.n_links,self.n_links)),
                                        dtype=self.dtype,
                                        name='nondiagonal_kernel_matrix'
                                        # constraint = tf.keras.constraints.NonNeg()
                                        )

    @property
    def kernel_matrix(self):
        self._kernel_matrix = tf.exp(self._diagonal_parameters)*tf.eye(self.n_links) \
                              + self._nondiagonal_parameters - tf.linalg.diag_part(self._nondiagonal_parameters)

        self._kernel_matrix = self.kernel_constraint(self._kernel_matrix)

        return self._kernel_matrix

    def call(self, input):
        return tf.matmul(input,self.kernel_matrix)

class MLP(PerformanceFunction):

    def __init__(self,
                 n_links,
                 dropout: float = 0.2,
                 alpha_relu = 0,
                 depth: int = 1,
                 polynomial_layer = None,
                 free_flow_traveltimes=None,
                 capacities=None,
                 kernel_constraint=None,
                 trainable = True,
                 *args,
                 **kwargs):
        super().__init__(*args, type='mlp', **kwargs)

        # initializer = tf.keras.initializers.Constant(1)
        self.free_flow_traveltimes = tf.constant(free_flow_traveltimes, self.dtype)
        self.capacities = tf.constant(capacities, self.dtype)
        self.bias_initializer = None
        self.kernel_initializer = None
        self.kernel_constraint = kernel_constraint
        self.depth = depth
        self.n_links = n_links
        self.polynomial_layer = polynomial_layer
        self.trainable_layer = trainable
        self.alpha_relu = alpha_relu
        self.polynomial_layer.alpha_relu = alpha_relu

        # if self.poly_order > 1:
        #     self.capacities = tf.repeat(tf.constant(self.capacities)[:, None], self.n_poly_features, 1)

        # self.weights_performance_function = tf.Variable(tf.ones(n_links,dtype = self.dtype))


        if free_flow_traveltimes is not None:
            self.bias_initializer = tf.constant_initializer(free_flow_traveltimes)
            # if capacities is not None:

        self.kernel_initializer = tf.constant_initializer(self.kernel_constraint.initial_values.numpy())

        self.bias_contraint = BiasConstraint(free_flow_traveltimes=self.free_flow_traveltimes)

    def build(self):

        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(None, self.n_links)),
            # tf.keras.layers.Dense(n_links, activation='relu'),
            # tf.keras.layers.Dropout(dropout),
            # tf.keras.layers.Dense(n_links),
            # tf.keras.layers.Dense(n_links, activation='relu', kernel_initializer = initializer),
            # tf.keras.layers.Dense(n_links, kernel_initializer=initializer, use_bias = False),
        ])

        if self.kernel_constraint.homogenous:
            # output_dim = 1
            # self.model.add(SingleWeightLayer(initial_value = 0.15))
            self.model.add(SingleWeightLayer(initial_value=1, name='link_flow_interaction_parameter'))

        else:

            self.model.add(KernelWeightLayer(initial_value_diagonal=1,
                                             initial_value_nondiagonal=1e-1,
                                             n_links = self.n_links,
                                             kernel_constraint= self.kernel_constraint,
                                             name='link_flow_interaction_matrix'))

        # else:
        #     output_dim = self.n_links
        #
        #     for i in range(self.depth):
        #         self.model.add(tf.keras.layers.Dense(output_dim,
        #                                              # activation='relu',
        #                                              name='link_flow_interaction_matrix',
        #                                              kernel_initializer=self.kernel_initializer,
        #                                              bias_initializer=self.bias_initializer,
        #                                              use_bias=False,
        #                                              # kernel_constraint = tf.keras.constraints.NonNeg()
        #                                              kernel_constraint=self.kernel_constraint,
        #                                              bias_constraint=self.bias_contraint
        #                                              # bias_constraint = tf.keras.constraints.NonNeg()
        #                                              ))
        #
        #     # self.model.add(tf.keras.layers.ReLU())
        #     self.model.add(tf.keras.layers.LeakyReLU(self.alpha_relu))
        #     # self.model.add(tf.keras.layers.PReLU())


        # if kernel_constraint is None or (self.kernel_constraint.diagonal is False):
        #     self.model.add(tf.keras.layers.ReLU())
        #     # tf.keras.layers.LeakyReLU(0.1)
        #     # tf.keras.layers.PReLU()\
        self.model.trainable = self.trainable_layer

        # Build parameters polynomial layer if it has been provided
        if self.polynomial_layer is not None and not self.polynomial_layer.built:
            self.polynomial_layer.build(n_links = self.n_links)

    def regularizer(self, regularizer):

        ''' Regularizer for the non-diagonal component of the kernel matrix'''

        if regularizer is None:
            regularizer = tf.keras.regularizers.L1(1)

        regularization_loss = 0

        if not self.kernel_constraint.diagonal and not self.kernel_constraint.homogenous:
            kernel = self.model.trainable_variables[-1]
            regularization_loss = regularizer(kernel - tf.eye(kernel.shape[0]) * tf.linalg.diag_part(kernel))

            # regularization_loss = tf.math.abs(tf.reduce_sum(
            #                           kernel - tf.eye(kernel.shape[0]) * tf.linalg.diag_part(kernel)))

        return regularization_loss

    def call(self, flows):

        """
        Flows are expected to be normalized by links' capacity

        :param flows:
        :return:
        """

        # return tf.clip_by_value(tf.cast(self.model(flows),self.dtype),
        #                         clip_value_min=self.free_flow_traveltimes,clip_value_max=self.free_flow_traveltimes*10))

        # if self.polynomial_layer is not None:
        #     flows = self.polynomial_layer.transform_flows(flows = flows, period_ids= self.period_ids)

        # input = self.model(tf.einsum("ijk,j -> ijk", flows, 1/self.capacities))

        # tf.repeat(tf.repeat(tf.constant(self.capacities)[:, None], self.n_poly_features, 1)[None],

        return self.free_flow_traveltimes * (1 + tf.cast(self.model(flows), self.dtype))

        # self.model.get_weights()
        # np.diagonal(self.model.get_weights())

    # def __call__(self, flows):
    #     return self.free_flow_traveltimes + tf.clip_by_value(self.weights_performance_function,1e-4, 100)*flows/self.capacities
    # return self.free_flow_traveltimes + self.weights_performance_function * flows/self.capacities
    # self.model.get_weights()


def bpr_function(flows, capacities, free_flow_traveltimes, alpha = 0.15, beta = 4):
    """
    Apply BPR function to input link flows
    """

    traveltimes = free_flow_traveltimes * (1 + alpha * tf.math.pow(flows / capacities, beta))
    # traveltimes = self.tt_ff* (1 + self.alpha * tf.math.pow(x / k, self.beta))

    return traveltimes

class BPR(PerformanceFunction):

    def __init__(self,
                 capacities=None,
                 free_flow_traveltimes=None,
                 dtype=None,
                 max_traveltime_factor = None,
                 *args,
                 **kwargs):
        super().__init__(dtype=dtype, type='bpr', max_traveltime_factor = max_traveltime_factor)

        self.parameters = pesuelogit.models.BPRParameters(*args, **kwargs)
        self.k = tf.constant(capacities, dtype=self.dtype)
        self.capacities = tf.constant(capacities, dtype=self.dtype)
        self.tt_ff = free_flow_traveltimes

        # Initialization of learnable variables

        self.initial_values = {}
        self.initial_values['alpha'] = self.parameters.parameters['alpha'].initial_value
        self.initial_values['beta'] = self.parameters.parameters['beta'].initial_value

    @property
    def alpha(self):
        if self._alpha is None:
            return float('nan')
        # Without the exponential trick, the estimation of alpha is more unstable.
        return tf.clip_by_value(tf.exp(self._alpha), 0, 10)
        # return tf.clip_by_value(tf.exp(self._alpha), 0, 4)

    @property
    def beta(self):

        if self._beta is None:
            return float('nan')

        return tf.clip_by_value(tf.exp(self._beta), 0, 10)
        # return tf.clip_by_value(tf.exp(self._beta),1,4.1)
        # return tf.clip_by_value(tf.exp(self._beta), self._epsilon, 10)

    def build(self):

        self._alpha = tf.Variable(
            # self.bpr.parameters['alpha'].initial_value,
            initial_value=np.log(self.initial_values['alpha']),
            # np.sqrt(self.bpr.parameters['alpha'].initial_value),
            trainable=self.parameters.parameters['alpha'].trainable,
            name='alpha',
            dtype=self.dtype)

        # self._parameters['alpha'] = self._alpha

        self._beta = tf.Variable(
            # self.bpr.parameters['beta'].initial_value,
            initial_value=np.log(self.initial_values['beta']),
            # np.sqrt(self.bpr.parameters['beta'].initial_value),
            trainable=self.parameters.parameters['beta'].trainable,
            name='beta',
            dtype=self.dtype)

        # self._parameters['beta'] = self._beta

    def traveltimes(self, x):
        """
        Apply BPR function to input link flows
        """

        traveltimes = self.tt_ff * (1 + self.alpha * tf.math.pow(x / self.k, self.beta))
        # traveltimes = self.tt_ff* (1 + self.alpha * tf.math.pow(x / k, self.beta))

        return traveltimes

    def call(self, flows):
        return tf.cast(self.traveltimes(flows), self.dtype)


class NESUELOGIT(PESUELOGIT):
    def __init__(self,
                 generation: GenerationParameters = None,
                 performance_function: PerformanceFunction = None,
                 input_shape=(256, 256, 3),
                 *args,
                 **kwargs):

        # self.build((None,) + input_shape)

        super().__init__(*args, **kwargs)

        self.dtype_int = tf.int32

        if self.dtype == tf.float64:
            self.dtype_int = tf.int64

        self._alpha = None
        self._beta = None

        self._observed_flow = None
        self._observed_traveltime = None
        self._kappa = None

        self.utility = copy.deepcopy(self.utility)
        self.od = copy.deepcopy(self.od)

        self.performance_function = copy.deepcopy(performance_function)
        self.generation = copy.deepcopy(generation)

        if self.generation is not None:
            self.generation.n_periods = self.n_periods

        self._equilibrium_stage = False
        self._node_data = None

        self.node_data_sorted = False

        self.o, self.d = np.array(self.od.ods).T

    # def mask_by_train_val_split(self, tensor, Y):
    #
    #     tf.math.count_nonzero(
    #
    #
    #     return tensor


    def update_predictions(self, X, period_dict = None, update_period_dict = False):

        self.set_period_ids(X.numpy(), period_dict, update_period_dict= update_period_dict)
        self.period_ids = X[:,:,-1]

        self._predicted_flow = self.predict_flow()
        self._predicted_traveltime = self.predict_traveltime()
        self._output_flow = self.compute_link_flows(X)
        self._input_flow = self.flows

    def compute_loss_metric(self,
                            metric=mse,
                            prefix_metric: str = 'mse_',
                            update_predictions=False,
                            X=None,
                            Y=None
                            ):
        '''
        Compute loss metric for travel time, link flow and equilibrium componets. The loss_function should be computed
        before

        prefix_metric: e.g. 'loss_'
        :return:
        '''

        if update_predictions:
            self.update_predictions(X)

            self._observed_traveltime, self._observed_flow = tf.unstack(self.mask_Y(Y), axis=-1)

        observed_traveltime = self._observed_traveltime
        observed_flow = self._observed_flow

        predicted_flow = self._predicted_flow
        predicted_traveltime = self._predicted_traveltime
        output_flow = self._output_flow
        input_flow = self._input_flow

        return {prefix_metric + 'flow': float(metric(actual=observed_flow, predicted=predicted_flow)),
                prefix_metric + 'traveltime': float(metric(actual=observed_traveltime, predicted=predicted_traveltime)),
                prefix_metric + 'equilibrium': float(metric(actual=input_flow, predicted=output_flow))
                }

    def compute_loss_metrics(self, metrics: Dict, X, Y):

        self.update_predictions(X)

        self._observed_traveltime, self._observed_flow = tf.unstack(self.mask_Y(Y), axis=-1)

        metrics_df = pd.DataFrame({})

        for name, metric in metrics.items():
            metrics_components = self.compute_loss_metric(metric=metric, prefix_metric='', X = X, Y = Y)
            metrics_df = pd.concat([metrics_df,
                                    pd.DataFrame({'component': metrics_components.keys(),
                                                  'value': metrics_components.values(),
                                                  'metric': name})])

        return metrics_df

    @property
    def flows(self):
        flows = tf.math.pow(self._flows, 2)

        if self.n_periods >= 1:
            flows = tf.experimental.numpy.take(flows, tf.cast(self.period_ids[:, 0], dtype=tf.int32), 0)

        return tf.cast(flows, self.dtype)

    def forward(self, X):

        self.period_ids = X[:, :, -1]

        # predicted_flow = self.compute_link_flows(X)
        # output_flow = predicted_flow
        # predicted_traveltimes = self.bpr_traveltimes(predicted_flow)

        # input_flow = tf.stop_gradient(self.flows)
        self._input_flow = self.flows

        self._output_flow = self.compute_link_flows(X)

        self._predicted_flow = self._input_flow  # tf.stop_gradient(self.predict_flow())
        # predicted_flow = self.predict_flow()
        # predicted_flow = output_flow

        # predicted_traveltimes = tf.stop_gradient(self.traveltimes())
        self._predicted_traveltime = self.traveltimes()

    def equilibrium_loss(self, input_flow, output_flow, observed_flow, Y, loss_metric = mse):

        """ 
        Equilibrium loss is normalized depending on the loss metric used to compute the link flow loss
        """

        # loss = loss_metric(actual=predicted_flow, predicted=output_flow)
        # loss = loss_metric(actual=self.flows, predicted=output_flow)
        if loss_metric == nrmse:
            # loss = rmse(actual=input_flow, predicted=output_flow)
            loss = rmse(actual=input_flow, predicted=output_flow)

            if Y is not None:
                loss /= tf.experimental.numpy.nanmean(observed_flow)

        elif loss_metric == zscore:

            loss = rmse(actual=input_flow, predicted=output_flow)
            # loss = mse(actual=input_flow, predicted=output_flow)

            if Y is not None:
                # loss /= np.nanstd(observed_flow)**2
                loss /= np.nanstd(observed_flow)

        elif loss_metric == nmse:
            # loss = rmse(actual=input_flow, predicted=output_flow)
            loss = mse(actual=input_flow, predicted=output_flow)

            if Y is not None:
                loss /= tf.experimental.numpy.nanmean(observed_flow)**2

        elif loss_metric == z2score:

            loss = mse(actual=input_flow, predicted=output_flow)
            # loss = mse(actual=input_flow, predicted=output_flow)

            if Y is not None:
                # loss /= np.nanstd(observed_flow)**2
                loss /= np.nanstd(observed_flow)**2

        else:
            loss = loss_metric(actual=input_flow, predicted=output_flow)

        return loss

    # @property
    def observed_traveltime(self, Y=None):

        if Y is None:
            return self.mask_observed_traveltime(self._observed_traveltime)
            # return float('nan')

        return self.mask_observed_traveltime(tf.unstack(Y, axis=-1)[0])

    # @property
    def observed_flow(self, Y=None):

        if Y is None:
            return  self.mask_observed_flow(self._observed_flow)
        #
        # return self.mask_observed_flow(tf.unstack(Y, axis=-1)[1])
        return self.mask_observed_flow(flow = tf.unstack(Y, axis=-1)[1])

    def mask_Y(self, Y):

        # observed_traveltime = self.mask_observed_traveltime(observed_traveltime)
        # observed_flow = self.mask_observed_flow(observed_flow)

        return mask_Y(Y = Y,
                      k = np.array([link.bpr.k for link in self.network.links]),
                      tt_ff = self.tt_ff,
                      D = self.D)

    def mask_observed_flow(self, flow):

        return mask_observed_flow(flow = flow, D = self.D)

    def mask_observed_traveltime(self, traveltime, k = None, k_threshold = 1e5):

        if k is None:
            k = np.array([link.bpr.k for link in self.network.links])
        #
        # mask1 = np.where((k >= k_threshold), float('nan'), 1)
        # mask2 = np.where((self.tt_ff == 0), float('nan'), 1)
        #
        # return mask2*mask1*traveltime

        return mask_observed_traveltime(traveltime = traveltime,
                                        k = k,
                                        tt_ff=self.tt_ff,
                                        k_threshold = k_threshold
                                        )

    def compute_regularization_loss(self, lambda_vals):

        loss = {}

        if self.performance_function.type == 'mlp' and lambda_vals['regularizer_kernel']>0:
            regularizer = tf.keras.regularizers.L1(lambda_vals['regularizer_kernel'])
            loss['regularizer_kernel'] = self.performance_function.regularizer(regularizer)

        # Only for model that has a generation stage
        if lambda_vals['regularizer_utility_ods']>0 and self.kappa is not None:

            regularizer = tf.keras.regularizers.L1(lambda_vals['regularizer_utility_ods'])
            loss['regularizer_utility_ods'] = regularizer(self.utility_ods)

        if lambda_vals['regularizer_od'] > 0:
            regularizer = tf.keras.regularizers.L2(lambda_vals['regularizer_od'])
            loss['regularizer_od'] = regularizer(self.q)

        return loss

    def loss_function(self,
                      X,
                      Y=None,
                      lambdas: Dict[str, float] = None,
                      loss_metric=None,
                      epoch=None
                      ):
        """
        Return a dictionary with keys defined as the different terms of the loss function

        # loss_metric = btcg_mse, mse

        """

        self.forward(X)

        # Normalize weights to one
        lambdas = {k: (v / np.sum(list(lambdas.values()))) for k, v in lambdas.items()}

        if Y is not None:
            # self._observed_traveltime, self._observed_flow = tf.unstack(self.mask_Y(Y), axis=-1)
            self._observed_traveltime, self._observed_flow = tf.unstack(Y, axis=-1)

        if loss_metric is None:
            # loss_metric = mnrmse
            loss_metric = mse

        lambdas_vals = {'traveltime': 0.0, 'od': 0.0, 'theta': 0.0, 'flow': 0.0, 'equilibrium': 0.0,
                        'bpr': 0, 'ntrips': 0, 'prop_od': 0}

        regularization_vals = {'regularizer_kernel': 0, 'regularizer_utility_ods': 0, 'regularizer_od': 0,}

        lambdas_vals = {**lambdas_vals, **regularization_vals}

        assert set(lambdas.keys()).issubset(lambdas_vals.keys()), 'Invalid key in loss_weights attribute'

        for attr, val in lambdas.items():
            lambdas_vals[attr] = val

        loss = dict.fromkeys(list(lambdas_vals.keys()) + ['total'], tf.constant(0, dtype=self.dtype))

        # Only regularize tgodlulpe model
        if self.key == 'tvgodlulpe':
            self.regularization_loss = self.compute_regularization_loss(lambdas_vals)
            loss = {**loss, **self.regularization_loss}


        # if self.od.features_generation:
        #     lambdas_vals['generation'] = 1e2
        #     loss['generation'] = mse(actual = compute_generated_trips(historic_od, ods = self.od.ods),
        #                              predicted = compute_generated_trips(self.q, ods = self.od.ods))

        # Equilibrium loss
        loss['equilibrium'] = self.equilibrium_loss(loss_metric = loss_metric,
                                                    Y = Y,
                                                    input_flow = self._input_flow,
                                                    output_flow = self._output_flow,
                                                    observed_flow = self._observed_flow)

        # np.nanmean(self._observed_traveltime)
        # np.nanmean(predicted_traveltimes)
        historic_od = self.q.numpy()
        historic_od[:] = np.nan

        # if lambdas_vals['od'] > 0 and self.od.historic_values is not None and self.q is not None:
        if self.od.historic_values is not None and self.q is not None:
            historic_od = tf.cast(self.od.historic_values_array, self.dtype)
            #historic_od = tf.cast(tf.stack(list(self.od.historic_values.values())), self.dtype)

            loss.update({
                'od': loss_metric(actual=historic_od, predicted=self.q),
                'ntrips': sse(actual=np.sum(self.q, axis=1),
                              predicted=self.od.total_trips_array) / historic_od.shape[1],
                'prop_od': loss_metric(actual=normalize_od(historic_od), predicted=normalize_od(self.q)),
            })
        # else:
        #     loss.update({'od': float('nan'), 'ntrips': float('nan'), 'prop_od': float('nan')})

        if Y is not None:
            loss.update({
                'flow': loss_metric(actual=self._observed_flow, predicted=self._predicted_flow),
                # 'flow': loss_metric(actual=self._observed_flow, predicted=output_flow),
                'traveltime': loss_metric(
                    # actual=tf.reduce_mean(self._observed_traveltime, axis = 0,keepdims=True),
                    actual=self._observed_traveltime,
                    predicted=self._predicted_traveltime),
                # 'theta': tf.reduce_mean(tf.norm(self.theta, 1)),
            }

                # mse(
                #     # actual=tf.reduce_mean(self._observed_traveltime, axis = 0,keepdims=True),
                #     actual=self._observed_traveltime,
                #     predicted=self._predicted_traveltime)
            )

        loss['total'] = tf.constant(0, self.dtype)

        lambdas = {k: tf.constant(v, name="lambda_" + k, dtype=self.dtype) for k, v in lambdas_vals.items()}

        # lambdas = {'od': tf.constant(lambdas_vals['od'], name="lambda_od", dtype=self.dtype),
        #            'theta': tf.constant(lambdas_vals['theta'], name="lambda_theta", dtype=self.dtype),
        #            'flow': tf.constant(lambdas_vals['flow'], name="lambda_flow", dtype=self.dtype),
        #            'traveltime': tf.constant(lambdas_vals['traveltime'], name="lambda_tt", dtype=self.dtype),
        #            'bpr': tf.constant(lambdas_vals['bpr'], name="lambda_bpr", dtype=self.dtype)
        #            }

        for key, val in lambdas_vals.items():
            # if any(list(map(lambda x: isinstance(val, x), [float, int]))):
            if val > 0:
                loss['total'] += lambdas[key] * tf.cast(loss[key], self.dtype)

        # Add prefix "loss_"
        loss = {'loss_' + k: v for k, v in loss.items()}

        return loss

    def path_utilities(self, V):
        # return self.path_size_correction(tf.einsum("ijk,kl -> ijl", V, self.D))
        return self.path_size_correction(tf.einsum("jk,kl -> jl", V, self.D))

    def path_size_correction(self, Vf):

        path_utilities = Vf

        if self.psc_factor == 0:
            return path_utilities
        else:
            return path_utilities + self.psc_factor * tf.cast(tf.math.log(tf.constant(
                isl.paths.compute_path_size_factors(D=self.network.D, paths_od=self.network.paths_od).flatten())),
                dtype=self.dtype)

    def project_parameters(self, parameters: Union[UtilityParameters, GenerationParameters], values):
        clips_min = []
        clips_max = []

        signs = parameters.signs

        if signs is not None:
            for feature in parameters.features:

                sign = signs.get(feature)

                if sign == '+':
                    clips_min.append(0)
                    clips_max.append(1e10)
                elif sign == '-':
                    clips_min.append(-1e10)
                    clips_max.append(0)
                else:
                    clips_min.append(-1e10)
                    clips_max.append(1e10)

            return tf.clip_by_value(values, clips_min, clips_max)

        return values

    @property
    def theta(self):

        return self.project_parameters(values=self._theta, parameters=self.utility)

    def get_parameters_estimates(self) -> pd.DataFrame:

        # TODO: extend for multiperiod theta and multilinks alpha, beta
        estimates = {}
        estimates.update(dict(zip(self.utility.features, np.mean(self.theta.numpy(), axis=0))))
        estimates['psc_factor'] = float(self.psc_factor.numpy())
        estimates['fixed_effect'] = np.mean(self.fixed_effect.numpy())

        if self.performance_function.type == 'bpr':
            estimates.update(dict(zip(['alpha', 'beta'], [np.mean(self.performance_function.alpha.numpy()),
                                                          np.mean(self.performance_function.beta.numpy())])))

        return pd.DataFrame(estimates, index=[0])

    def get_parameters_values(self, model=None):

        if model is None:
            model = self

        link_flows = model.flows
        if model.n_periods > 1:
            link_flows = tf.concat([tf.expand_dims(tf.reduce_mean(link_flows, axis=0), axis=0)
                                    for link_flows in model.split_link_flows_by_period(model.flows)], axis=0)

        initial_values = {'flows': link_flows,
                          'theta': model.theta,
                          # 'alpha': model.performance_function.alpha,
                          # 'beta': model.performance_function.beta,
                          'q': model.q,
                          'fixed_effect': model.fixed_effect,
                          'psc_factor': model.psc_factor,
                          }

        # TODO: add values from generation and distribution. May remove plain q parameter

        return initial_values

    def mask_predicted_traveltimes(self, tt, k, k_threshold=1e5, max_factor = None):
        """

        :param tt:
        :param k:
        :param k_threshold:
        :param max_factor: constraint the maximum value that travel times can take
        :return:
        """

        # mask1 = np.where((k >= k_threshold) | (self.tt_ff == 0), 1, 0)
        # mask2 = np.where((k >= k_threshold), 1, 0)
        # mask3 = np.where((self.tt_ff == 0), 1, 0)
        #
        # return (1-mask1)*(1-mask2)*self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta)) + (1-mask3)*mask2*self.tt_ff

        mask1 = np.where((k >= k_threshold), 1, 0)
        mask2 = np.where((self.tt_ff == 0), 1, 0)

        predicted_traveltimes = (1 - mask2) * ((1 - mask1) * tt + mask1 * self.tt_ff)

        if max_factor is not None:
            # TODO: Define factor according to speed instead of travel time by using link lengths
            max_traveltimes = tf.cast(max_factor * self.tt_ff * tf.ones_like(tt), dtype=self.dtype)
            return tf.math.minimum(predicted_traveltimes, max_traveltimes)

        return predicted_traveltimes

        # return self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

    def traveltimes(self, flows = None):
        """ Return tensor variable associated to endogenous travel times (assumed dependent on link flows)"""

        if self.performance_function.type == 'bpr':
            if flows is None:
                flows = self.flows

        elif self.performance_function.type == 'mlp':

            if flows is None:
                flows = tf.math.pow(self._flows, 2)

            flows = self.performance_function.polynomial_layer.transform_flows(
                flows,
                capacities = self.performance_function.capacities,
                period_ids = self.period_ids,
                n_periods = self.n_periods)

        traveltimes = self.performance_function.call(flows)

        traveltimes = self.mask_predicted_traveltimes(tt=traveltimes,
                                                      k=np.array([link.bpr.k for link in self.network.links]),
                                                      max_factor = self.performance_function.max_traveltime_factor)

        return traveltimes

        # return tf.experimental.numpy.take(traveltimes,tf.cast(self.period_ids[:,0], dtype = tf.int32),0)

    def predict_traveltime(self):

        return self.traveltimes()

    def predict_flow(self):
        return self.flows

    def class_membership_probabilities(self):

        n_classes = self.utility.n_classes

        raise NotImplementedError

    def link_utilities(self, X):

        """ TODO: Make the einsum operation in one line"""

        self.period_ids = X[:, :, -1]
        theta = tf.cast(tf.experimental.numpy.take(self.theta, tf.cast(self.period_ids[:, 0], self.dtype_int), 0),
                        dtype=self.dtype)
        # theta = self.theta

        # if tf.rank(theta) == 1:
        #     # return tf.einsum("ijkl,l -> ijk", X, self.theta[1:])+ self.theta[0]*self.traveltimes() + self.fixed_effect
        #     return theta[0] * self.traveltimes() + tf.einsum("ikl,l -> ik", X[:, :, :-1], theta[1:]) + self.fixed_effect

        return self.traveltimes() * tf.expand_dims(theta[:, 0], 1) \
            + tf.einsum("ijk,ik -> ij", X[:, :, :-1], theta[:, 1:]) + self.fixed_effect

    def path_probabilities_sparse(self, vf, normalization=False):
        ''' Sparse version. Computation time is roughly the same than non-sparse version but it does not require
        to store the matrix C which has dimensions n_paths X n_paths
        tf.sparse.reduce_max has no gradient registered, thus this op is ignored from the backprop:
        https://www.tensorflow.org/api_docs/python/tf/stop_gradient

        #TODO: Optimize repetition of M_sparse matrix over days and hours dimensions

        '''

        M_sparse = tf.cast(tf.sparse.from_dense(self.network.M), self.dtype)
        # M_sparse = tf.sparse.concat(0, [tf.sparse.expand_dims(M_sparse, 0)] * vf.shape[1])
        M_sparse = tf.sparse.concat(0, [tf.sparse.expand_dims(M_sparse, 0)] * vf.shape[0])

        indices = M_sparse.indices

        V = tf.sparse.SparseTensor(indices=indices,
                                   # values = tf.exp(tf.reshape(vf,-1)),
                                   values=tf.reshape(vf, [-1]),
                                   # dense_shape=(vf.shape[0], vf.shape[1], *self.M.shape)
                                   dense_shape=[vf.shape[0], self.M.shape[0], self.M.shape[1]]
                                   )

        if normalization:
            # normalized_values = V.values - tf.reshape(
            #     tf.einsum("ijk,kl -> ijl", tf.stop_gradient(tf.sparse.reduce_max(V, axis=3)), self.M), -1)

            # V = tf.sparse.SparseTensor(indices=indices, values=tf.exp(normalized_values),
            #                            dense_shape=(vf.shape[0], vf.shape[1], *self.M.shape))

            normalized_values = V.values - tf.reshape(
                tf.einsum("jk,kl -> jl", tf.stop_gradient(tf.sparse.reduce_max(V, axis=2)), self.M), [-1])

            V = tf.sparse.SparseTensor(indices=indices,
                                       values=tf.exp(normalized_values),
                                       dense_shape=[vf.shape[0], self.M.shape[0], self.M.shape[1]]
                                       )

        else:
            V = tf.sparse.map_values(tf.exp, V)

        return tf.reshape(V.values, vf.shape) / tf.einsum("jk,kl -> jl", tf.sparse.reduce_sum(V, axis=2), self.M)

    @property
    def kappa(self):
        if self._kappa is not None:
            return self.project_parameters(values=self._kappa, parameters=self.generation)
        else:
            return None
        # return self._kappa

    @property
    def fixed_effect_generation(self):
        # TODO: May mask this vector of parameters by only allowing nodes that report generated trips to be different than 0.
        # May also mask node_data
        return self._fixed_effect_generation

    def log(self, value, margin=1e-10):
        return tf.cast(tf.math.log(value + margin), dtype=self.dtype)

    @property
    def node_data(self):

        if self._node_data is None:
            return None

        if not self.node_data_sorted:
            # node_keys = pd.DataFrame({'key': [node for node, _ in self.od.ods]}).drop_duplicates()
            # node_keys = pd.DataFrame({'key': [node.key for node in self.network.nodes]})
            # node_keys = pd.DataFrame({'key': list(range(0, len([node.key for node in self.network.nodes])))})
            node_keys = pd.DataFrame({'key': list(range(0, len([self.network.nodes[node_idx].key for node_idx in self.od.nodes])))})

            self._node_data = pd.merge(node_keys, self._node_data, on='key')

            self.node_data_sorted = True

        features = []

        for feature in self.generation.features:
            if feature in self._node_data.columns:
                features.append(feature)
            else:
                raise IndexError

        if len(features) > 0:

            # Imputation
            data = self._node_data[features].values

            return data

        else:
            return 0

    def load_node_data(self, data: pd.DataFrame):
        self._node_data = data

    def pretrain_generation_weights(self, one_hot_encoding=False, pvalues=True):
        '''

        :param one_hot_encoding: include one fixed effect for every location that can generate trips
        :return:
        '''

        print("\nPretraining generation weights\n")

        enc = OneHotEncoder(handle_unknown='ignore')

        # y = self.od.compute_generated_trips().numpy().flatten()

        y = self.generation.historic_g.numpy()

        if one_hot_encoding:
            fixed_effect = enc.fit_transform(np.arange(0, self.node_data.shape[0])[:, np.newaxis]).toarray()
        else:  # Single intercept
            fixed_effect = np.ones((self.node_data.shape[0], 1))

        X = np.concatenate([self.node_data, fixed_effect], axis=1)

        kappa = np.ones((self.n_periods,len(self.generation.features)))
        fixed_effect_generation = np.ones_like(self._fixed_effect_generation.numpy())

        for period in range(self.n_periods):

            if one_hot_encoding:
                Xt, yt = X, y[period]
            else:
                # Remove observations where the number of generated trips is zero.
                idxs_non_zeros = (y[period] != 0)
                Xt, yt = X[idxs_non_zeros, :], y[period][idxs_non_zeros]

            reg = LinearRegression(fit_intercept=False).fit(X = Xt, y = yt)

            if pvalues:
                estimation = sm.OLS(yt, sm.add_constant(Xt)).fit()
                print(f'period {period} -> p-values kappa:',
                      dict(zip(self.generation.features, np.round(estimation.pvalues, 4))))

            coefs = reg.coef_

            for i in range(self._kappa.shape[1]):
                kappa[period,i] = coefs[i]

            fixed_effect_generation[period] = coefs[len(self.generation.features):]

        self._kappa.assign(kappa)

        self._fixed_effect_generation.assign(tf.convert_to_tensor(fixed_effect_generation, self.dtype))

    def mask_generation_nodes(self, g):
        """
        Mask nodes where the number of generated trips is zero.

        :return:
        """

        if self.generation.historic_g is None:
            return g

        return g * tf.cast(self.generation.historic_g > 0, dtype=self.dtype)

    @property
    def g(self):

        '''
        Trip generation: generate a vector with the total number of trips at each location using a linear regression

        :return: a vector with the total number of trips at each location
        '''

        if self.node_data is None or self._kappa is None:
            g = self.fixed_effect_generation

        else:
            g = self.fixed_effect_generation + tf.matmul(self.kappa, self.node_data.T)

        # return g
        return tf.nn.relu(self.mask_generation_nodes(g))

    @property
    def utility_ods(self):
        '''
        Using origin and destination specific effects can decrease the number of parameters but it may ncrease training
        time due to the 'take' operation.
        :return:
        '''

        utility = self._fixed_effect_od

        if self._fixed_effect_origin.trainable and self._fixed_effect_destination.trainable:
            utility = tf.experimental.numpy.take(self._fixed_effect_origin, self.o, axis=1) \
                      + tf.experimental.numpy.take(self._fixed_effect_destination, self.d, axis=1)

        elif self._fixed_effect_destination.trainable:
            utility = tf.experimental.numpy.take(self._fixed_effect_destination, self.d, axis=1)

        return utility

    @property
    def phi(self):
        '''
        Trip distribution: Apply logit model to obtain probability of choosing a destination from any given location.
        :param o: vector with the total number of trips at each location
        :return: a sparse matrix of probabilities with dimensions |N| x |N|

        Note: Currently fixed effects are assumed to be period specific, which makes attribute specific parameters to be
        not identifiable
        '''

        # Stack a set of sparse L matrix for every period
        L_sparse = tf.sparse.concat(0, [tf.sparse.expand_dims(self.od.L_sparse, 0)] * self.n_periods)

        # tf.sparse.to_dense(self.od.L_sparse)
        # L = tf.sparse.to_dense(L_sparse)[0]

        indices = L_sparse.indices

        # Utility of od-pairs (TODO: add destination specific features)
        v = self.utility_ods

        V = tf.sparse.SparseTensor(indices=indices,
                                   values=tf.reshape(v, [-1]),
                                   dense_shape=(self.n_periods, len(self.od.nodes), len(self.od.nodes))
                                   )

        normalized_values = V.values - tf.stop_gradient(tf.reshape(
            tf.repeat(tf.sparse.reduce_max(V, axis=2), tf.sparse.reduce_sum(self.od.L_sparse, axis=1), axis=1), [-1]))

        # normalized_values = V.values

        V = tf.sparse.SparseTensor(indices=indices,
                                   values=tf.exp(normalized_values),
                                   dense_shape=(self.n_periods, len(self.od.nodes), len(self.od.nodes)))

        phi = tf.reshape(V.values, v.shape) / \
              tf.repeat(tf.sparse.reduce_sum(V, axis=2), tf.sparse.reduce_sum(self.od.L_sparse, axis=1), axis=1)

        return phi

    @property
    def q(self):

        if self.generation is None:

            if self._q is None:
                return None

            # Old representation of OD matrix in dense form
            return tf.math.pow(self._q, 2)

        # g = tf.stop_gradient(tf.repeat(self.g, 1, axis=1))

        # TODO: review choice of axis to sum on
        # g = tf.repeat(self.g, tf.sparse.reduce_sum(self.od.L_sparse, axis=1), axis=1)
        g = tf.experimental.numpy.take(self.g, self.o, axis=1)

        # tf.reduce_sum(np.array([[1, 2], [3, 4]]), axis=1)

        # tf.sparse.to_dense(self.od.L_sparse)

        q = g * self.phi

        # tf.reduce_sum(g, axis = 1)

        return q

    @property
    def Q(self, average = False):

        if average and tf.rank(self.q) == 2:
            q = tf.reduce_mean(self.q, axis=0)

            return tf.SparseTensor(
                indices=self.triplist,
                values=q,
                dense_shape=(self.n_periods, self.n_nodes, self.n_nodes)
            )

        return tf.sparse.to_dense(self.od.get_sparse_tensor(values = self.q))


    def create_tensor_variables(self, keys: Dict[str, bool] = None,
                                trainables: Dict[str, bool] = None,
                                initial_values: Dict[str, bool] = None
                                ):

        if keys is None:
            keys = dict.fromkeys(['q', 'theta', 'psc_factor', 'fixed_effect', 'alpha', 'beta',
                                  'fixed_effect_distribution', 'fixed_effect_generation'], True)

        trainables_defaults = {'flows': self.endogenous_flows,
                               'q': self.od.trainable,
                               'theta': self.utility.trainables,
                               'kappa': self.generation.trainables if self.generation is not None else False,
                               'fixed_effect_generation': False,
                               'fixed_effect_distribution': False,
                               }

        if trainables is not None:
            for k, v in trainables_defaults.items():
                if k not in trainables.keys():
                    trainables[k] = trainables_defaults[k]
        else:
            trainables = trainables_defaults

        initial_values_defaults = {
            'flows': tf.constant(tf.zeros([self.n_periods, self.n_links], dtype=self.dtype)),
            # 'flows': tf.constant(tf.zeros([self.n_links], dtype=self.dtype)),
            # 'q': tf.cast(self.od.initial_values_array(), self.dtype),
            'q':  tf.cast(self.od.initial_values_array(), self.dtype) if len(self.od.initial_value.shape) == 1 else
            tf.cast(self.od.initial_value, self.dtype),
            'theta': tf.cast(self.utility.initial_values_array(self.utility.features), self.dtype),
            'kappa': tf.cast(self.generation.initial_values_array(
                self.generation.features), self.dtype) if self.generation is not None else None,
            'psc_factor': self.utility.initial_values['psc_factor'],
            'fixed_effect': tf.constant(self.utility.initial_values['fixed_effect'], dtype=self.dtype),
        }

        if initial_values is not None:
            for k, v in initial_values_defaults.items():
                if k not in initial_values.keys():
                    initial_values[k] = initial_values_defaults[k]

        else:
            initial_values = initial_values_defaults

        # Link specific effect (act as an intercept)
        # if self.endogenous_flows:
        self._flows = tf.Variable(
            initial_value=tf.math.sqrt(initial_values['flows']),
            # initial_value=initial_values['flows'],
            # initial_value=tf.constant(tf.zeros([self.n_timepoints,self.n_links]), dtype=self.dtype),
            trainable=trainables['flows'],
            name='flows',
            dtype=self.dtype)

        # Log is to avoid that parameters are lower and equal than zero.
        # Sqrt is to avoid that that parameters are strictly lower than zero

        if keys.get('theta', False):

            theta = []

            for i, feature in enumerate(self.utility.features):
                theta_i = tf.Variable(
                    # initial_value= initial_values['theta'][feature],
                    initial_value=tf.cast(initial_values['theta'][:, i], self.dtype),
                    trainable=trainables['theta'][feature],
                    name=feature,
                    dtype=self.dtype)

                theta.append(theta_i)

            # # TODO: Temporarily, the attribute of travel time defines if theta is trainable.
            self._theta = tf.Variable(tf.stack(theta, axis=1),
                                      trainable=trainables['theta']['tt'],
                                      name='theta')

            self._parameters['theta'] = self._theta

        if keys.get('psc_factor', False):
            # Initialize the psc_factor in a value different than zero to generate gradient
            self._psc_factor = tf.Variable(initial_value=initial_values['psc_factor'],
                                           trainable=trainables['theta']['psc_factor'],
                                           name=self.utility.parameters['psc_factor'].key,
                                           dtype=self.dtype)

            self._parameters['psc_factor'] = self._psc_factor

        if keys.get('fixed_effect', False):
            # Link specific effect (act as an intercept)
            self._fixed_effect = tf.Variable(
                initial_value=initial_values['fixed_effect'],
                trainable=trainables['theta']['fixed_effect'],
                name=self.utility.parameters['fixed_effect'].key,
                dtype=self.dtype)

            self._parameters['fixed_effect'] = self._fixed_effect

        if self.generation is None:
            self._q = tf.Variable(initial_value=np.sqrt(initial_values['q']),
                                  trainable=trainables['q'],
                                  name=self.od.key,
                                  dtype=self.dtype)

            self._parameters['q'] = self._q

        else:

            # Generation features

            initial_values['fixed_effect_generation'] = self.generation.initial_values['fixed_effect']

            # initial_values['fixed_effect_generation'] \
            #      = self.generation.initial_values['fixed_effect'] * tf.ones((self.n_periods, len(self.od.nodes)))


            # Link specific effect (act as an intercept)
            self._fixed_effect_generation = tf.Variable(
                initial_value=tf.cast(initial_values['fixed_effect_generation'], dtype=self.dtype),
                trainable=self.generation.trainables['fixed_effect'],
                name='fixed_effect_generation',
                dtype=self.dtype)

            kappa = []

            for i, feature in enumerate(self.generation.features):
                # self._parameters['fixed_effect_generation'] = self._fixed_effect_generation
                kappa.append(
                    tf.Variable(
                        # initial_value= tf.cast(initial_values['theta'][:,i],self.dtype),
                        # trainable=trainables['theta'][feature],
                        initial_value=tf.cast(initial_values['kappa'][:, i], self.dtype),
                        trainable=self.generation.trainables[feature],
                        name=feature,
                        dtype=self.dtype)
                )

            # TODO: Temporarily, the first feature in generation features dict defines if kappa is trainable.
            # self._kappa = None

            if len(self.generation.features) > 0:
                self._kappa = tf.Variable(tf.stack(kappa, axis=1),
                                          # trainable=self.generation.trainables['fixed_effect'],
                                          trainable=self.generation.trainables[self.generation.features[0]],
                                          name='kappa')

            # Distribution features
            initial_values['fixed_effect_od'] \
                = tf.constant(0, shape=tf.TensorShape(self.od.initial_value.shape), dtype=self.dtype)

            self._fixed_effect_od = tf.Variable(
                initial_value=initial_values['fixed_effect_od'],
                # trainable=False,
                trainable=self.generation.trainables.get('fixed_effect_od', False),
                name='fixed_effect_distribution',
                dtype=self.dtype)

            initial_values['fixed_effect_origin'] \
                = tf.constant(0, shape=tf.TensorShape((self.n_periods, len(self.network.nodes))), dtype=self.dtype)

            self._fixed_effect_origin = tf.Variable(
                initial_value=initial_values['fixed_effect_origin'],
                # trainable=True,
                trainable=self.generation.trainables.get('fixed_effect_origin', False),
                name='fixed_effect_origin',
                dtype=self.dtype)

            self._fixed_effect_destination = tf.Variable(
                initial_value=initial_values['fixed_effect_origin'],
                # trainable=True,
                trainable=self.generation.trainables.get('fixed_effect_destination', False),
                name='fixed_effect_destination',
                dtype=self.dtype)

            self._parameters['fixed_effect_generation'] = self._fixed_effect_od

        # Create parameters of performance function
        self.performance_function.build()

    def compute_link_flows(self, X):

        self.period_ids = X[:, :, -1]

        link_flows = self.link_flows(
            self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))

        # if tf.rank(link_flows) == 2:
        #     return self.split_link_flows_period(link_flows)

        # return tf.reduce_mean(link_flows,axis = 0)

        return link_flows

    def compute_travel_times(self, X):

        # self.period_ids = X[:, :, -1]

        link_flows = self.link_flows(
            self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))

        # if tf.rank(link_flows) == 2:
        #     return self.split_link_flows_period(link_flows)

        # return tf.reduce_mean(link_flows,axis = 0)

        return link_flows

    def path_flows(self, pf):
        # TODO: Test and try to combine the einsums if possible to avoid ifelse clause

        # if tf.rank(self.q) == 1:
        #     return tf.einsum("ij,i, lj -> lj", self.M, self.q, pf)
        #     # return tf.einsum("ij,i, klj -> klj", self.M, self.q, pf)
        #     # return tf.einsum("j,lij -> lij", tf.einsum("ij,i-> j", self.M, self.q), pf)

        # todo: have a single einsum, e.g. return tf.einsum("ij,ki, lkj -> lij", self.M, self.q, pf)
        # return tf.einsum("ij, lij -> lij", tf.einsum("ij,ki-> kj", self.M, self.q), pf)
        od_flows = tf.einsum("ij,ki-> kj", self.M, self.q)
        od_flows = tf.experimental.numpy.take(od_flows, tf.cast(self.period_ids[:, 0], dtype=tf.int32), 0)

        return tf.einsum("ij, ij -> ij", od_flows, pf)

    # def call(self, inputs, *args, **kwargs):
    def call(self, X, inputs=None):
        """
        It allows saving model
        """
        # a = inputs
        # if node_data is None:
        #     node_data = self.node_data

        return X

        # return self.loss_function(*args, **kwargs)

    # @tf.function(
    #     input_signature=[tf.TensorSpec(shape=(None, None, None))]
    # )
    def predict(self, X: tf.Tensor, pretrain_generation_weights = False,
                dtype=tf.float32, period_dict: Dict = None, **kwargs):

        """
        X: tensor of link features of dimension (n_daus, n_hours, n_links, n_features)
        period_dict: mapping between original periods ids and internal periods ids

        return tensor of dimension (n_timepoints, n_links) with a forward pass
        """

        if self.generation is not None:
            self.generation._pretrain_generation_weights = pretrain_generation_weights

        # sum = np.sum(X[:, :, -1])
        self.compute_equilibrium(X, **kwargs)
        # assert sum == np.sum(X[:, :, -1])

        self.update_predictions(X, period_dict)

        predicted_flow = self._input_flow

        predicted_traveltime = self.traveltimes()

        return tf.cast(tf.concat([predicted_traveltime[:, :, np.newaxis], predicted_flow[:, :, np.newaxis]], axis=2),
                       dtype)

    def compute_equilibrium(self, X, optimizer, epochs, *args, **kwargs):

        print('\nComputing gradient based equilibrium')

        kwargs.update({'equilibrium_stage': True, 'epochs': {'learning': 0, 'equilibrium': epochs},
                       'optimizers': {'learning': None, 'equilibrium': optimizer},
                       'X_train': X}
                      )

        return self.fit(*args, **kwargs)

    def set_period_ids(self, X: np.array, period_dict=None, update_period_dict=True):

        period_ids = X[:, :, -1].copy()

        if period_dict is None:
            period_ids = X[:, :, -1]
            period_keys = np.unique(period_ids[:, 0])

            # This dictionary can map the original period ids into the indices of the array used internally by the model
            period_dict = dict(zip(list(map(int, np.sort(period_keys))), range(0, len(period_keys))))

        for i, j in period_dict.items():
            period_ids[period_ids == i] = j

        if update_period_dict:
            self.period_dict = period_dict
            # self.period_ids = period_ids

        return tf.concat([X[:, :, :-1], tf.expand_dims(period_ids, 2)], axis=2)

    def setup_period_ids(self,
                         X_train: tf.Tensor,
                         Y_train: tf.Tensor = None,
                         X_val: tf.Tensor = None,
                         Y_val: tf.Tensor = None,
                         node_data=None,
                         period_dict=None,
                         ):

        if node_data is not None:
            self.load_node_data(node_data)

        X_train, Y_train = [copy.copy(tf.cast(i, self.dtype)) if i is not None else None for i in [X_train, Y_train]]

        X_train = self.set_period_ids(X=X_train.numpy(), period_dict=period_dict)
        self.period_ids = X_train[:, :, -1]

        if X_val is not None and Y_val is not None:
            X_val, Y_val = map(lambda x: copy.copy(tf.cast(x, self.dtype)), [X_val, Y_val])
            X_val = self.set_period_ids(X=X_val.numpy())

        # Also update period ids values in historic values of OD and total trips dictionary
        # (#TODO: may move this code under od class)

        if self.od.historic_values is not None and isinstance(self.od.historic_values, dict):
            new_historic_values = {}
            for period_id, od in self.od.historic_values.items():
                for k, v in self.period_dict.items():
                    if k == period_id:
                        new_historic_values[v] = od

            self.od.historic_values = new_historic_values

        if self.od.total_trips is not None:

            new_total_trips = {}
            for period_id, ntrips in self.od.total_trips.items():
                for k, v in self.period_dict.items():
                    if k == period_id:
                        new_total_trips[v] = ntrips

            self.od.total_trips = new_total_trips

        return X_train, Y_train, X_val, Y_val

    def reset_weights(self):
        self.load_weights(self._filepath_weights)

    def build(self, input_shape=None):

        # TODO: should require period_dict based on input dictionary

        if not self.built:
            self.create_tensor_variables()
            self.built = True

            # TODO: create folder output/models if it does not exist
            self._model_id = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{self.key}_{self.network.key}"
            self._filepath_weights = f"output/models/{self._model_id}.h5"

        # Save weights

        # self._filepath_weights = f'output/models/{self.key}_{self.network.key}.h5'
        # self.save_weights(self._filepath_weights)

        # self._filepath_weights = f'output/models/{self.key}_{self.network.key}.h5'
        # self.save_weights(self._filepath_weights)

    # def compute_relative_gap(self):
    #
    #     relative_gap = (
    #             tf.norm(link_flow - self.flows, 1) / (normalizer * tf.norm(self.flows, 1))).numpy()

    def fit(self,
            X_train: tf.Tensor,
            Y_train: tf.Tensor = None,
            X_val: tf.Tensor = None,
            Y_val: tf.Tensor = None,
            node_data: pd.DataFrame = None,
            optimizers=None,
            loss_weights: Dict[str, float] = None,
            epochs: Dict[str, int] = None,
            threshold_relative_gap: float = float('inf'),
            loss_metric=None,
            momentum_equilibrium=1,
            pretrain_link_flows = False,
            equilibrium_stage=False,
            alternating_optimization=False,
            relative_losses=True,
            epochs_print_interval: Dict[int, int] = None,
            batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """ It assumes the last column of tensors X_train and X_val contains period ids. The remaining
        columns contains the exogenous features of each link"""

        print('\nModel training')

        if not self.built:
            # with tf.compat.v1.variable_scope('model'):
            self.build()

        X_train, Y_train, X_val, Y_val = self.setup_period_ids(X_train=X_train,
                                                               Y_train=Y_train,
                                                               X_val=X_val,
                                                               Y_val=Y_val,
                                                               node_data=node_data)

        # Mask observed flow and travel time


        if self.generation is not None and self.generation.pretrain_generation_weights:
            self.pretrain_generation_weights()
            # self.generation._pretrain_generation_weights = False

        # Initialization of endogenous travel times and flows
        if pretrain_link_flows:

            predicted_flow = self.compute_link_flows(X_train)

            if self.n_periods == 1:

                self._flows.assign(tf.math.sqrt(tf.expand_dims(tf.reduce_mean(predicted_flow, axis=0), axis=0)))

            else:
                predicted_flow = tf.concat([tf.expand_dims(tf.reduce_mean(link_flows, axis=0), axis=0)
                                            for link_flows in self.split_link_flows_by_period(predicted_flow)], axis=0)

                self._flows.assign(tf.math.sqrt(predicted_flow))

            print("\nLink flows and travel times were pretrained with single pass of traffic assignment")

        # Pretrained polynomial weights
        if self.performance_function.type == 'mlp' and self.performance_function.polynomial_layer._pretrain_weights:
                self.performance_function.polynomial_layer.pretrain_weights(
                    flows = tf.math.pow(self._flows, 2),
                    free_flow_traveltimes=self.tt_ff,
                    capacities=self.performance_function.capacities)

        # Split data by batch
        if batch_size is None:
            batch_size = X_train.shape[0]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0]).batch(batch_size)
        # train_dataset = train_dataset.batch(batch_size)

        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        # val_dataset = val_dataset.batch(batch_size)

        # Initial Losses
        # train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights)['loss_total']
        # val_loss = self.loss_function(X=X_val, Y=Y_val, loss_weights=loss_weights)['total']
        estimates = [self.get_parameters_estimates()]

        # MSE is keep here regardless the selected loss metric so it is printed the true loss
        train_loss = {k: float(v) for k, v in
                      self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights, loss_metric=mse).items()}

        train_losses = [train_loss]

        if Y_train is not None:
            train_losses = [{**train_losses[0], **self.compute_loss_metric(metric=mape, prefix_metric='mape_')}]

        # # Set inital travel time loss with free flow travel times
        # train_losses[-1]['loss_traveltime'] = mse(actual=self.observed_traveltime(Y=Y_train),
        #                                  predicted=[link.bpr.tf for link in self.network.links])

        val_losses = [{k: 0 * v for k, v in train_losses[0].items()}]

        if X_val is not None and Y_val is not None:
            val_loss = {k: float(v)
                        for k, v in self.loss_function(X=X_val, Y=Y_val, lambdas=loss_weights, loss_metric=mse).items()}
            val_losses = [{**val_loss, **self.compute_loss_metric(metric=mape, prefix_metric='mape_')}]
            # val_losses[-1]['loss_traveltime'] = mse(actual=self.observed_traveltime(Y=Y_val),
            #                               predicted=[link.bpr.tf for link in self.network.links])

        if epochs_print_interval is None:
            epochs_print_interval = {'learning': 1, 'equilibrium': 1}

        if 'equilibrium' not in epochs.keys():
            epochs['equilibrium'] = 0

        if 'learning' not in epochs.keys():
            epochs['learning'] = 0

        # Coverage of observed link flows and travel times
        coverages = {'train': float('nan'), 'val': float('nan')}
        coverages_count = {'train': float('nan'), 'val': float('nan')}

        for group, data in {'train': Y_train, 'val': Y_val}.items():
            if data is not None:
                coverages_count[group] \
                    = np.floor(tf.reduce_mean(tf.reduce_sum(tf.cast(~tf.math.is_nan(data), self.dtype), 1), 0))

                coverages[group] \
                    = np.round((tf.reduce_sum(tf.reduce_sum(tf.cast(~tf.math.is_nan(data), self.dtype), 1), 0) /
                                (tf.cast(tf.size(data), self.dtype) / 2)).numpy(), 2)

        # relative_gaps = [float('nan')]
        relative_gaps = []
        convergence = False

        epoch = 0
        t0 = time.time()

        total_epochs = epochs['learning']

        if equilibrium_stage:
            total_epochs += epochs['equilibrium']

        current_stage = 'equilibrium'

        if epochs['learning']>0:
            current_stage = 'learning'

        # Print benchmark based on mean reported in training data
        if Y_val is not None:
            print("Benchmark metrics using mean in training data to make predictions in the validation set: ")
            with pd.option_context('display.float_format', '{:0.2g}'.format):
                print('\n')
                print(compute_benchmark_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2_score}, Y_ref=Y_train, Y=Y_val))

        # Training loop
        while not convergence:

            if epoch == 0:
                print('\nhyperparameters loss function:', loss_weights)
                print(f"\nnumber of periods: {self.n_periods}, batch size: {batch_size}, "
                      f"threshold relative gap: {threshold_relative_gap}"
                      # f"learning rate: "
                      , end="")

                # Training iset
                print(f"\ntraining set -> ", end='', flush=True)
                print(f"timepoints: {X_train.shape[0]}, "
                      f"obs [t x]: {coverages_count['train']}, "
                      f"coverage [t x]: {coverages['train']}"
                      , end="")
                if X_val is not None:
                    print(f"\nvalidation set -> ", end='', flush=True)
                    print(f"timepoints: {X_val.shape[0]}, "
                          f"obs [t x]: {coverages_count['val']}, "
                          f"coverage [t x]: {coverages['val']}")

                print('')

            if not convergence:
                # self.period_ids = X_train[:, :, -1]
                path_flows = self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X_train))))
                link_flow = self.link_flows(path_flows)
                # relative_x = float(np.nanmean(np.abs(tf.divide(link_flow,self.flows) - 1)))

                if epoch >= 0:
                    normalizer = 1
                    if self.flows.shape[0] < link_flow.shape[0]:
                        normalizer = int(link_flow.shape[0] / self.flows.shape[0])

                    # We now use the definition of relative residual
                    relative_gap = (
                            tf.norm(link_flow - self.flows, 1) / (normalizer * tf.norm(self.flows, 1))).numpy()
                    relative_gaps.append(relative_gap)

                # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")

            if (epoch == total_epochs) or (abs(relative_gaps[-1]) < threshold_relative_gap and
                                           ((current_stage == 'equilibrium') or (epoch == epochs['learning']))):
                convergence = True


            if epoch == 0 and epochs['learning'] > 0:
                print(f"Learning stage: {epochs['learning']} epochs")

            if (epoch % epochs_print_interval['learning'] == 0) or (epoch == 1) or (epoch == epochs['learning'] + 1) or \
                    convergence or (equilibrium_stage and epoch % epochs_print_interval['equilibrium'] == 0):

                # print(f"\nEpoch: {epoch}/{total_epochs}, ", end="")

                # self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights, loss_metric=mse)

                print(f"\n{epoch}/{total_epochs}: "
                      f"train mse={float(train_losses[-1]['loss_total']):0.2g}, ",
                      end='', flush=True)

                if Y_train is not None:
                    print(f"train mse traveltime={float(train_losses[-1]['loss_traveltime']):0.2g}, "
                          f"train mape traveltime={float(train_losses[-1]['mape_traveltime']):0.1f}",
                          f"train mse flow={float(train_losses[-1]['loss_flow']):0.2g}, "
                          f"train mape flow={float(train_losses[-1]['mape_flow']):0.1f}, ",
                          end='', flush=True)

                if Y_val is not None:
                    print(f"val mse={float(val_losses[-1]['loss_total']):0.2g}, "
                          f"val mse traveltime={float(val_losses[-1]['loss_traveltime']):0.2g}, "
                          f"val mape traveltime={float(val_losses[-1]['mape_traveltime']):0.1f}",
                          f"val mse flow={float(val_losses[-1]['loss_flow']):0.2g}, "
                          f"val mape flow={float(val_losses[-1]['mape_flow']):0.1f}, ",
                          end='', flush=True)

                print(# f"train_loss bpr={float(train_loss['loss_bpr'].numpy()):0.2g}, "
                    # f"val_loss bpr={float(val_loss['loss_bpr'].numpy()):0.2g}, "
                    # f"theta = {np.round(np.unique(self.theta.numpy(),axis =0),3)}, "
                    f"theta = {np.round(np.mean(self.theta.numpy(), axis=0), 3)}, "
                    f"avg rr = {np.array(compute_rr(self.get_parameters_estimates().to_dict(orient='records')[0])):0.2f}, "
                    # f"psc_factor = {self.psc_factor.numpy()}, "
                    f"avg theta fixed effect = {np.mean(self.fixed_effect):0.2g}, "
                    # f"avg abs diff demand ={np.nanmean(np.abs(self.q - self.historic_od(self.q))):0.2g}, ",end = '')
                    f"loss prop od={train_losses[-1]['loss_od']:0.2g}, "
                    # f"loss ntrips={train_losses[-1]['loss_ntrips']:0.2g}, "
                    f"total trips={np.array2string(np.round(np.sum(self.q, axis=1)), formatter={'float': lambda x: f'{x:.2e}'})}, ",
                    end='', flush=True)
                if self.performance_function.type == 'bpr':
                    print(f"avg alpha={np.mean(self.performance_function.alpha.numpy()):0.2g}, "
                          f"avg beta={np.mean(self.performance_function.beta.numpy()):0.2g}, ", end='', flush=True)
                elif self.performance_function.type == 'mlp':
                    # poly_weights = np.array2string(self.performance_function.weights[0].numpy().flatten(),
                    #                                formatter={'float': lambda x: f'{x:.1e}'})
                    poly_weights = np.array2string(
                        np.mean(self.performance_function.polynomial_layer.poly_weights.numpy(),axis = 0),
                        formatter={'float': lambda x: f'{x:.1e}'})
                    print(f"polynomial weights: {poly_weights}, ", end='', flush=True)

                if self._kappa is not None:
                    print(f"kappa = {np.round(np.mean(self.kappa.numpy(), axis=0), 3)}, ", end='', flush=True)

                if train_losses[-1].get('loss_equilibrium', False):
                    print( f"lambda eq={loss_weights['equilibrium']:0.2g}, "
                           f"relative gap={relative_gaps[-1]:0.2g}, ", end='', flush=True)

                    print(f"train equilibrium loss={train_losses[-1]['loss_equilibrium']:0.2g}, ", end='', flush=True)

                    if Y_val is not None:
                        print(f"val equilibrium loss={val_losses[-1]['loss_equilibrium']:0.2g}, ", end='', flush=True)

                print(f"time:{time.time() - t0: 0.1f}")

                t0 = time.time()

            if epochs['learning'] > 0 and not convergence and \
                    ((equilibrium_stage and (epoch == (epochs['learning'])) and epochs['learning'] != 0) or
                     (epochs['learning'] == 0 and epoch == 0)):

                current_stage = 'equilibrium'
                print(f"\nEquilibrium stage: {epochs['equilibrium']} epochs")

            if not convergence:

                # if epoch == epochs['learning'] and epochs['equilibrium']>0 and alternating_optimization:

                # selected_trainable_variables = trainable_variables

                loss_weights['equilibrium'] = loss_weights['equilibrium'] / momentum_equilibrium

                if epoch < epochs['learning']:
                    # Learning part

                    trainable_variables = self.trainable_variables
                    if alternating_optimization is True:
                        trainable_variables = [j for j in self.trainable_variables if j.name.split(':')[0] != 'flows']

                    for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):
                        with tf.GradientTape() as tape:
                            train_loss = \
                                self.loss_function(X=X_batch_train, Y=Y_batch_train, lambdas=loss_weights,
                                                   loss_metric=loss_metric)['loss_total']

                        grads = tape.gradient(train_loss, trainable_variables)

                        optimizers['learning'].apply_gradients(zip(grads, trainable_variables))

                if alternating_optimization or (epoch >= epochs['learning'] and equilibrium_stage):

                    loss_weights_equilibrium = loss_weights.copy()

                    if current_stage == 'equilibrium':
                        iterations = epochs['equilibrium']
                        loss_weights_equilibrium = dict(
                            zip(loss_weights.keys(), [0] * len(loss_weights.keys())))
                        loss_weights_equilibrium['equilibrium'] = 1

                    elif current_stage == 'learning' and alternating_optimization:
                        # loss_weights_equilibrium = loss_weights.copy() # This is like coordinate descent
                        iterations = 1


                    # for i in range(iterations):
                    trainable_variables = [j for j in self.trainable_variables if
                                           j.name.split(':')[0] == 'flows']

                    assert len(trainable_variables) == 1

                    # if alternating_optimization:
                    #
                    #     with tf.GradientTape() as tape:
                    #         train_loss = \
                    #             self.loss_function(X=X_train, Y=Y_train,
                    #                                lambdas=loss_weights_equilibrium,
                    #                                loss_metric=loss_metric)['loss_total']
                    #
                    #     grads = tape.gradient(train_loss, trainable_variables)
                    #
                    #     optimizers['equilibrium'].apply_gradients(zip(grads, trainable_variables))
                    #
                    # else:

                    for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):
                        with tf.GradientTape() as tape:
                            train_loss = self.loss_function(X=X_batch_train, Y=Y_batch_train,
                                                            lambdas=loss_weights_equilibrium,
                                                            loss_metric=loss_metric)['loss_total']

                        grads = tape.gradient(train_loss, trainable_variables)

                        optimizers['equilibrium'].apply_gradients(zip(grads, trainable_variables))

                # Store losses and estimates
                train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights, loss_metric=mse,
                                                epoch=epoch)

                if Y_train is not None:
                    train_loss = {**train_loss, **self.compute_loss_metric(metric=mape, prefix_metric='mape_')}

                # train_loss = {**train_loss, **self.compute_loss_metric(metric=mape, prefix_metric='mape')}

                val_loss = {k: 0 * v for k, v in train_loss.items()}

                if X_val is not None and Y_val is not None:
                    val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=loss_weights, loss_metric=mse, epoch=epoch)
                    val_loss = {**val_loss, **self.compute_loss_metric(metric=mape, prefix_metric='mape_')}

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                estimates.append(self.get_parameters_estimates())

                epoch += 1

            if convergence and (abs(relative_gaps[-1]) < threshold_relative_gap):
                print(f'\nRelative gap threshold of {threshold_relative_gap} was achieved in {current_stage} stage')

            if self.column_generator is not None:
                # TODO: Column generation (limit the options to more fundamental ones)
                self.column_generator.generate_paths(theta=self.theta,
                                                     network=self.network
                                                     )

            # TODO: Path set selection (confirm if necessary)

        # Store key results
        train_losses_df = pd.concat([pd.DataFrame([losses_epoch], index=[0]).astype(float).assign(epoch=epoch)
                                     for epoch, losses_epoch in enumerate(train_losses)])

        val_losses_df = pd.concat([pd.DataFrame([losses_epoch], index=[0]).astype(float).assign(epoch=epoch)
                                   for epoch, losses_epoch in enumerate(val_losses)])

        # Replace equilibirum loss in first epoch with the second epoch, to avoid zero relative loss due to the
        # initialization strategy for link flows when initial thetas are zero
        train_losses_df.loc[train_losses_df['epoch'] == 0, 'loss_equilibrium'] \
            = train_losses_df.loc[train_losses_df['epoch'] == 1, 'loss_equilibrium'].copy()
        val_losses_df.loc[val_losses_df['epoch'] == 0, 'loss_equilibrium'] \
            = val_losses_df.loc[val_losses_df['epoch'] == 1, 'loss_equilibrium'].copy()

        # train_losses_df.loc[train_losses_df['epoch'] == 0,'loss_od'] \
        #     = train_losses_df.loc[train_losses_df['epoch'] == 1,'loss_od'].copy()
        # val_losses_df.loc[val_losses_df['epoch'] == 0, 'loss_od'] \
        #     = val_losses_df.loc[val_losses_df['epoch'] == 1, 'loss_od'].copy()

        train_results_df = pd.concat([train_losses_df.reset_index(drop=True),
                                      pd.concat(estimates, axis=0).reset_index(drop=True)], axis=1). \
            assign(relative_gap=relative_gaps)

        val_results_df = val_losses_df.reset_index(drop=True)

        # train_losses_df['generalization_error'] = train_generalization_errors
        # val_losses_df['generalization_error'] = val_generalization_errors

        if not alternating_optimization:
            train_results_df['stage'] = 'learning'
            val_results_df['stage'] = 'learning'

        if relative_losses:
            # Compution of relative losses
            # losses_columns = [column for column in train_losses_df.keys() if column not in ["loss_equilibrium"]]
            losses_columns = [column for column in train_losses_df.keys() if 'loss_' in column or 'epoch' == column]
            train_results_df[losses_columns] = normalized_losses(train_results_df[losses_columns])
            # .assign(loss_equilibrium = train_losses_df['loss_equilibrium'])
            val_results_df[losses_columns] = normalized_losses(val_results_df[losses_columns])
            # .assign(loss_equilibrium = val_losses_df['loss_equilibrium'])

        return train_results_df, val_results_df



def create_inference_model(reference_model, creation_method,  **kwargs):

    reference_model.save_weights(reference_model._filepath_weights)

    if not kwargs.get('n_periods', False):
        kwargs['n_periods'] = reference_model.n_periods

    if not kwargs.get('historic_g', False):
        kwargs['historic_g'] = reference_model.generation.historic_g

    if not kwargs.get('network', False):
        kwargs['network'] = reference_model.network

    new_model, parameters = creation_method(**kwargs)

    # for name, parameter in {**parameters['utility_parameters'].parameters,
    #                         **parameters['generation_parameters'].parameters,
    #                         **parameters['performance_function'].parameters.parameters,
    #                         **parameters['od_parameters'].parameters
    #                         }.items():
    #     parameter.trainable = False

    new_model.build()

    new_model.load_weights(reference_model._filepath_weights)

    return new_model

def train_val_split_by_links(Y: np.array, val_size=0) -> (np.ndarray, np.ndarray):
    """
    Return training and validation set of the output features, setting nan for observations that
    do not belong to the set. Observation with nan are not considerd for the train_val split.

    :param Y: observed data with shape (N, n_links, source)
    :param val_size: value between 0,1 representing the proportion of values assigned to validation set.

    :return: two numpy arrays with training and validation set
    """
    _, n_links, _ = Y.shape

    # if val_size<1:
    #     val_size = int(val_size*n_links)

    n_links = Y.shape[1]

    Y_train = Y.copy()
    Y_val = Y.copy()

    if val_size == 0:
        return Y_train, Y_val

    # tt_idxs, flow_idxs = (np.count_nonzero(~np.isnan(y), axis=0) > 0).T

    # val_idxs = tt_idxs[np.random.choice(np.arange(len(tt_idxs)), val_size, replace=False)]
    # tt_idxs[list(set(range(len(tt_idxs))) - set(val_idxs))]
    # for i, idxs in {0: tt_idxs, 1: flow_idxs}.items():

    for i in range(Y_train.shape[2]):
        # Identify links that have at least one non-missing observation during the time horizon
        observed_links_idxs = np.where((np.count_nonzero(~np.isnan(Y[:, :, i]), axis=0) > 0))[0]
        val_idxs = np.random.choice(observed_links_idxs, int(val_size * len(observed_links_idxs)), replace=False)
        train_idxs = np.array(list(set(observed_links_idxs) - set(val_idxs)))

        Y_train[:, np.array(list(val_idxs)), i] = float('nan')
        Y_val[:, np.array(list(train_idxs)), i] = float('nan')

    return Y_train, Y_val

def make_kfold(y: tf.Tensor, n_splits=2, random_state: int = None, shuffle=False):
    """

    :param X:
    :param y: vector of data associated to a single data source (N x n_links), e.g. travel times
    :param n_splits:
    :return:
    """

    if random_state is not None:
        shuffle = True

    dtype = y.dtype

    def gen():

        N, n_links = y.shape

        observed_links_idxs = np.where((np.count_nonzero(~np.isnan(y), axis=0) > 0))[0]

        for train_idxs, val_idxs in KFold(n_splits, random_state=random_state, shuffle=shuffle).split(
                observed_links_idxs):
            y_train = y.numpy().copy()
            y_val = y.numpy().copy()

            y_train[:, np.array(list(set(range(n_links)) - set(observed_links_idxs[train_idxs])))] = float('nan')
            y_val[:, np.array(list(set(range(n_links)) - set(observed_links_idxs[val_idxs])))] = float('nan')

            yield tf.cast(y_train, dtype), tf.cast(y_val, dtype)

    return tf.data.Dataset.from_generator(gen, (dtype, dtype))


# @timeit
def train_kfold(model: NESUELOGIT,
                X: tf.Tensor,
                Y: tf.Tensor,
                n_splits: int,
                random_state=None,
                *args,
                **kwargs):
    traveltime_folds = make_kfold(y=Y[:, :, 0], n_splits=n_splits, random_state=random_state)
    flow_folds = make_kfold(y=Y[:, :, 1], n_splits=n_splits, random_state=random_state)

    X_train = X.numpy().copy()
    X_val = X.numpy().copy()

    # train_results, val_results = [], []

    optimizers = kwargs['optimizers']

    metrics_df = pd.DataFrame()

    model.build()
    # Create folder kfold if it does not exist
    filepath = f"output/models/kfold/{model._model_id}.h5"
    model.save_weights(filepath)

    parameters_df = pd.DataFrame({})

    for i, traveltime_fold, flow_fold in zip(range(1, n_splits + 1), traveltime_folds, flow_folds):
        print(f'\nFold {i}/{n_splits}')

        traveltime_train, traveltime_val = traveltime_fold
        flow_train, flow_val = flow_fold

        Y_train = tf.concat([traveltime_train[:, :, np.newaxis], flow_train[:, :, np.newaxis]], axis=2)
        Y_val = tf.concat([traveltime_val[:, :, np.newaxis], flow_val[:, :, np.newaxis]], axis=2)

        Y_train = model.mask_Y(Y=Y_train)
        Y_val = model.mask_Y(Y=Y_val)

        X_train, X_val, Y_train, Y_val = [tf.cast(i, model.dtype) for i in [X_train, X_val, Y_train, Y_val]]

        # Compute metrics_df before training
        model.load_weights(filepath)
        with block_output(show_stdout=False, show_stderr=False):
            model.fit(X_train, Y_train, X_val, Y_val, *args,
                      **{**kwargs, **{'epochs': {'training': 0, 'equilibrium': 0}}})

        for X_cur, Y_cur, dataset_label in [(X_train, Y_train, 'training'), (X_val, Y_val, 'validation')]:
            cur_metrics_df = model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2_score},
                                                        X=X_cur, Y=Y_cur).assign(dataset=dataset_label)

            metrics_df = pd.concat([metrics_df, cur_metrics_df.assign(fold=i, stage='initial')])

        # Compute metrics_df after training
        model.load_weights(filepath)

        train_results_df, val_results_df = model.fit(X_train, Y_train, X_val, Y_val, *args, **kwargs)
        relative_gap = train_results_df['relative_gap'].values[-1]


        # Store parameters values
        for period in range(model.theta.shape[0]):

            theta_dict = dict(zip(model.utility.features, list(model.theta[period].numpy())))
            theta_dict['vot'] = compute_rr(theta_dict)
            parameters_df = pd.concat([parameters_df,
                                       pd.DataFrame({'parameter': theta_dict.keys(),
                                                     'value': theta_dict.values(),
                                                     'period': period,
                                                     'fold': i,
                                                     'group': 'utility'})])

            if model.generation is not None and model.kappa is not None:
                parameters_df = pd.concat([parameters_df,
                                           pd.DataFrame({'parameter': model.generation.features,
                                                         'value': list(model.kappa[period].numpy()),
                                                         'period': period,
                                                         'fold': i,
                                                         'group': 'generation'})])

        for X_cur, Y_cur, dataset_label in [(X_train, Y_train, 'training'), (X_val, Y_val, 'validation')]:
            cur_metrics_df = model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2_score},
                                                        X=X_cur, Y=Y_cur).assign(dataset=dataset_label)

            metrics_df = pd.concat([metrics_df, cur_metrics_df.assign(fold=i, stage='final')])

        # Add benchmark that is set as the average values of the observed measurements in the training set
        metrics_df = pd.concat([metrics_df,
                                compute_benchmark_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2_score},
                                                          Y_ref=Y_train, Y=Y_train).assign(dataset='training',
                                                                                           fold = i, stage = 'benchmark')])
        metrics_df = pd.concat([metrics_df,
                                compute_benchmark_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2_score},
                                                          Y_ref=Y_train, Y=Y_val).assign(dataset='validation',
                                                                                         fold = i, stage = 'benchmark')])

        metrics_df['relative_gap'] = relative_gap

        for optimizer in optimizers.values():
            for var in optimizer.variables():
                var.assign(tf.zeros_like(var))

    os.remove(filepath)

    metrics_df['n_splits'] = n_splits
    metrics_df['model'] = model._model_id


    parameters_df['n_splits'] = n_splits
    parameters_df['model'] = model._model_id

    metrics_df = metrics_df.reset_index().drop('index', axis=1)
    parameters_df = parameters_df.reset_index().drop('index', axis=1)

    return metrics_df, parameters_df
    # print(metrics.groupby(['dataset','component', 'metric'])['value'].aggregate(['mean','std']))
    # metrics.groupby(['dataset', 'component', 'metric'])['value'].aggregate({'mean': np.mean})

    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


def regularization_kfold(target_metric: str, target_component: str, loss_weights: List[Dict], **kwargs):

    """

    :param target_metric:
    :param target_component:
    :param loss_weights:
    :param kwargs:
    :return: dataframe with the average target metric for each loss component and metrics for the optimal
    regulatization weights
    """

    min_loss = float('inf')
    min_idx = None

    metrics_kfold_dfs = []
    parameters_kfold_dfs = []
    metrics_dfs = pd.DataFrame({})

    kwargs['model'].build()
    # Create folder kfold if it does not exist
    filepath = f"output/models/kfold/{kwargs['model']._model_id + '_regularization'}.h5"
    kwargs['model'].save_weights(filepath)

    for i, weights in enumerate(loss_weights):
        kwargs['loss_weights'] = weights
        kwargs['model'].load_weights(filepath)

        print(f"\nReplicate: {i+1}/{int(len(loss_weights))}\n")
        print(f"weights:  {weights}")

        metrics_kfold_df, parameters_kfold_df = train_kfold(**kwargs)

        # Stores metrics
        metrics_kfold_dfs.append(metrics_kfold_df)
        parameters_kfold_dfs.append(parameters_kfold_df)

        # Store mse of the loss components and the corresponding loss weights
        metrics_df = metrics_kfold_df[(metrics_kfold_df.metric == 'mse') & (metrics_kfold_df.stage == 'final')].\
            groupby(['component', 'dataset'])[['value']].mean().reset_index()

        for k,v in dict(zip(['lambda_' + i for i in weights.keys()], weights.values())).items():
            metrics_df[k] = v

        metrics_df['relative_gap'] = metrics_kfold_df['relative_gap']
        metrics_df['replicate'] = i

        metrics_dfs = pd.concat([metrics_dfs, metrics_df])

        # metrics_dfs = pd.concat([metrics_df, pd.DataFrame(
        #     {**dict(zip(['lambda_' + i for i in weights.keys()], weights.values())),
        #      **row.to_dict(orient='records')[0],
        #      **{'relative_gap': metrics_kfold_df['relative_gap']}}, index = [0]
        # )
        #                         ])

        # Compute minimum loss
        avg_loss = metrics_kfold_df[
            (metrics_kfold_df.metric == target_metric) & (metrics_kfold_df.component == target_component) & (
                        metrics_kfold_df.stage == 'final') & (metrics_kfold_df.dataset == 'validation')]['value'].mean()

        if avg_loss < min_loss:
            min_loss = avg_loss
            min_idx = i

        print(f'\nResults replicate: {i+1}, target metric: {target_metric}, '
              f'average {target_metric} {target_component}: {float(avg_loss):0.2g}, best replicate: {min_idx+1}')

    # metrics_df = metrics_df.reset_index().drop('index', axis=1)

    os.remove(filepath)

    return metrics_dfs, loss_weights[min_idx], metrics_kfold_dfs[min_idx], parameters_kfold_dfs[min_idx]


def compute_benchmark_metric(metric=mse,
                             prefix_metric: str = 'mse_',
                             Y_ref=None,
                             Y=None
                             ):
    '''
    Compute benchmark with the historical mean of Y_ref. Y contains the true valesu of the
    observations of travel time and link flow

    prefix_metric: e.g. 'loss_'
    :return:
    '''

    predicted_traveltime, predicted_flow \
        = tf.unstack((np.nanmean(Y_ref.numpy().reshape(-1, Y_ref.shape[-1]), 0) * np.ones_like(Y)), axis =-1)

    observed_traveltime, observed_flow = tf.unstack(Y, axis=-1)

    return {prefix_metric + 'flow': float(metric(actual=observed_flow, predicted=predicted_flow)),
            prefix_metric + 'traveltime': float(metric(actual=observed_traveltime, predicted=predicted_traveltime)),
            # prefix_metric + 'equilibrium': float(metric(actual=input_flow, predicted=output_flow))
            }

def compute_benchmark_metrics(metrics: Dict, Y_ref, Y):

    metrics_df = pd.DataFrame({})

    for name, metric in metrics.items():
        metrics_components = compute_benchmark_metric(Y_ref = Y_ref, Y = Y, metric=metric, prefix_metric='')
        metrics_df = pd.concat([metrics_df,
                                pd.DataFrame({'component': metrics_components.keys(),
                                              'value': metrics_components.values(),
                                              'metric': name})])

    return metrics_df


def compute_baseline_predictions_kfold(X: tf.Tensor, y: tf.Tensor, coordinates, n_splits: int, seed: int):
    """

    :param X: inputs for regression, independent features
    :param y: travel time or counts vector
    :param coordinates: point coordinates associated to links, e.g., centroid of links
    :param n_splits: splits for kfold
    :param _SEED: seed for reproducibility
    :return:
    """

    metrics_kfold = pd.DataFrame({})

    folds = make_kfold(y= y, n_splits=n_splits, random_state=seed)

    X = X.numpy()

    isinstance(y.numpy(), tf.Tensor)

    for i, fold in zip(range(1, n_splits + 1), folds):
        # print(f'Fold {i}/{n_splits}')

        y_train, y_val = fold

        y_train, y_val = [i.numpy().flatten() for i in [y_train, y_val]]

        train_idxs = np.where(~np.isnan(y_train.flatten()))[0]
        val_idxs = np.where(np.isnan(y_train.flatten()))[0]

        with block_output(show_stdout=False, show_stderr=False):
            metrics_df = compute_baseline_predictions(X_train=X[train_idxs, :],
                                                      X_val=X[val_idxs, :],
                                                      y_train=y_train[train_idxs],
                                                      y_val=y_val[val_idxs],
                                                      coordinates_train= coordinates[train_idxs, :],
                                                      coordinates_val= coordinates[val_idxs, :],
                                                      metric = mape
                                                      )

        metrics_df = metrics_df.assign(fold=i, metric='mape', n_splits=n_splits)

        metrics_kfold = pd.concat([metrics_kfold, metrics_df])

    return metrics_kfold

def compute_baseline_predictions(X_train, X_val, y_train, y_val, coordinates_train, coordinates_val,
                                 metric = mape):
        """

        Models include historical mean, regression, ordinary krigging and krigging regression

        :param X: matrix with features used to make predictions (|S| * |A| x |D|)
        :param y: vector of either travel time or counts (|S| x |A| x 1)
        :param coordinates: matrix with lon, lat coordinates of links (|A| x 2)
        :return:

        where S is the set of samples, A is the set of links and D is the set of exogenous features

        """

        # compute_benchmark_metric_model(metric=mape, Y_actual=Y_train, Y_pred=Y_pred)
        # p = X
        # x = coordinates
        # # target = train_df['counts'].values[:,np.newaxis]
        # target = y # train_df[target_label].values#[:, np.newaxis]
        #
        # # identify nan
        # nonnan_idx = np.where(np.any(~np.isnan(target), axis=1))[0]
        #
        # p = p[nonnan_idx]
        # x = x[nonnan_idx]
        # target = target[nonnan_idx]

        # p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
        #     p, x, target, test_size=0.1#, random_state=42
        # )

        p_train, p_test, x_train, x_test, target_train, target_test \
            = X_train, X_val, coordinates_train, coordinates_val, y_train, y_val

        # Get rid of y values with nan observations
        metrics_dict = {}

        # 0) Historical mean
        y_pred_historical_mean = np.nanmean(target_train)

        metrics_dict['historical_mean'] = float(metric(target_test, y_pred_historical_mean * np.ones_like(target_test)))

        # print('MAPE Historical mean', ))
        # print('Median APE', median_absolute_percentage_error(target_test, y_pred_historical_mean))

        # 1) Linear regression

        lr_model = LinearRegression(copy_X=True, fit_intercept=True)

        lr_model.fit(x_train, target_train)

        metrics_dict['linear regression'] = float(metric(actual = target_test, predicted = lr_model.predict(x_test)))

        # print('MAPE Linear regression', float(mape(actual = target_test, predicted = lr_model.predict(x_test))))
        # print('Median APE', median_absolute_percentage_error(target_test, lr_model.predict(x_test)))

        # m_rk.regression_model.score(p_test, target_test)

        # 2) Ordinary kriging
        OK = OrdinaryKriging(
            x_train[:, 0],
            x_train[:, 1],
            target_train.flatten(),
            # variogram_model="gaussian",
            # variogram_model= 'linear',
            variogram_model='exponential', # In line with Selby and Kockelman 2013
            verbose=False,
            enable_plotting=False,
            # nlags = 20
        )

        try:
            pred_okrigging, ss = OK.execute("points", x_test[:, 0], x_test[:, 1])

            metrics_dict['ordinary krigging'] = float(metric(target_test.flatten(), pred_okrigging))
        except:
            metrics_dict['ordinary krigging'] = float('nan')

        # print('MAPE Ordinary Krigging:', float(mape(target_test.flatten(), pred_okrigging)))
        # print('Median APE', median_absolute_percentage_error(target_test, pred_okrigging))

        # # 3) Universal kriging (too slow and perform similar than ordinary krigging)
        # UK = UniversalKriging(
        #     x_train[:, 0],
        #     x_train[:, 1],
        #     target_train.flatten(),
        #     # variogram_model="linear",
        #     variogram_model='exponential', # In line with Selby and Kockelman 2013
        #     verbose=False,
        #     drift_terms=["regional_linear"]
        # )
        #
        # # TODO: Make predictions in test set
        #
        # pred_ukrigging, ss = UK.execute("points", x_test[:, 0], x_test[:, 1])
        #
        # print('MAPE Universal Krigging:', float(mape(target_test.flatten(), pred_ukrigging)))
        # # print('Median APE', median_absolute_percentage_error(target_test, pred_ukrigging))

        # 4) Regression kriging

        # train_gdf.drop(['date'],axis = 1).to_file(f'{main_dir}/examples/scripts/output/fresno_network.shp',
        #                                           driver='ESRI Shapefile')
        lr_model = LinearRegression(copy_X=True, fit_intercept=True)

        m_rk = RegressionKriging(regression_model=lr_model, n_closest_points=10, variogram_model = 'exponential')
        # m_rk = RegressionKriging(regression_model=lr_model)

        # rk = RegressionKriging(regression_model='linear', variogram_model='spherical')


        with block_output(show_stdout=False, show_stderr=False):
            m_rk.fit(p_train, x_train, target_train.flatten())

        prediction_reg_krigging = m_rk.predict(p_test, x_test)

        metrics_dict['krigging regression'] = float(metric(target_test.flatten(), prediction_reg_krigging))

        # print('MAPE Regression krigging', float(mape(target_test.flatten(), prediction_reg_krigging)))

        return pd.DataFrame({'model':metrics_dict.keys(), 'value': metrics_dict.values()})

        # print('MAPE Regression krigging in sample', float(mape(target_train.flatten(), m_rk.predict(p_train, x_train))))
        # print('Median APE', median_absolute_percentage_error(target_train, m_rk.predict(p_train, x_train)))

        # # Geographical Weighted Regression (not working well due to matrix inversion issues)
        # # https://deepnote.com/@carlos-mendez/PYTHON-GWR-and-MGWR-71dd8ba9-a3ea-4d28-9b20-41cc8a282b7a
        #
        # # Some columns are dropped as the matrix becomes singular
        # X_train_gwr = p_train[:,0:-5]
        # X_test_gwr = p_test[:,0:-5]
        # # X_train_gwr = np.array([])
        # # X_test_gwr = np.array([])
        #
        # coords = list(zip(x_train[:, 0], x_train[:, 1]))
        #
        # gwr_selector = Sel_BW(coords, target_train, X_train_gwr, multi=True, constant=True)
        #
        # gwr_bw = gwr_selector.search(multi_bw_min=[2], multi_bw_max=[10])
        #
        # gwr_model = GWR(x_train, target_train, X_train_gwr, gwr_bw)
        #
        # gwr_results = gwr_model.fit()
        #
        # # gwr_results.summary()
        #
        # # train_df['gwr_R2'] = gwr_results.localR2
        #
        # gwr_predictions = gwr_model.predict(x_test, X_test_gwr, gwr_results.scale, gwr_results.resid_response).predictions
        #
        # print('MAPE Geographical Weighted Regression', float(mape(target_test, gwr_predictions)))
        # # print('Median APE', median_absolute_percentage_error(target_test, gwr_predictions))

        # # Multi Geographical Weighted Regression (NOTE: prediction is not implemented)
        # # https://deepnote.com/@carlos-mendez/PYTHON-GWR-and-MGWR-71dd8ba9-a3ea-4d28-9b20-41cc8a282b7a
        #
        # # Some columns are dropped as the matrix becomes singular
        # X_train_mgwr = p_train[:, 0:-4]
        # X_test_mgwr = p_test[:, 0:-4]
        #
        # coords = list(zip(x_train[:, 0], x_train[:, 1]))
        #
        # mgwr_selector = Sel_BW(coords, target_train[:, np.newaxis], X_train_mgwr, multi=True, constant=True)
        #
        # mgwr_bw = mgwr_selector.search(multi_bw_min=[2], multi_bw_max=[10])
        #
        # mgwr_model = MGWR(x_train, target_train[:, np.newaxis], X_train_gwr, mgwr_selector, constant=True)
        #
        # mgwr_results = mgwr_model.fit()
        #
        # # mgwr_results.summary()
        #
        # # train_df['gwr_R2'] = gwr_results.localR2
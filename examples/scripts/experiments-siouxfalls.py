import copy
import os
import time
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import nesuelogit.models

plt.ioff()

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
import seaborn as sns
from sklearn import preprocessing
from datetime import datetime

# To run code on CPU (two times much faster than GPU)
tf.config.set_visible_devices([], 'GPU')

# Modules from pesuelogit
from pesuelogit.visualizations import plot_heatmap_demands, plot_convergence_estimates
from pesuelogit.models import compute_rr, compute_insample_outofsample_error
from pesuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from pesuelogit.etl import get_design_tensor, get_y_tensor, add_period_id

# Internal modules
from nesuelogit.models import NESUELOGIT, ODParameters, UtilityParameters, BPR, MLP, KernelConstraint, \
    GenerationParameters, train_val_split_by_links, train_kfold, compute_generated_trips, compute_generation_factors, \
    create_inference_model, compute_benchmark_metrics, PolynomialLayer

from nesuelogit.visualizations import plot_predictive_performance, plot_metrics_kfold, \
    plot_top_od_flows_periods, plot_utility_parameters_periods, plot_mlp_performance_functions, plot_flow_vs_traveltime

from nesuelogit.metrics import mse, btcg_mse, mnrmse, mape, r2, nrmse, zscore, z2score

_PLOTS = True

# # Seed for reproducibility
_SEED = 2021
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

# TODO: Add a module with tests
# Test for tensorflow metal
# import tensorflow as tf
#
# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_val, y_val) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizers="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)


## Build network
network_name = 'SiouxFalls'
# network_name = 'Eastern-Massachusetts'
tntp_network = build_tntp_network(network_name=network_name)

## Read OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q=Q)

Q_historic = isl.factory.random_disturbance_Q(tntp_network.Q.copy(), sd=np.mean(tntp_network.Q) * 0.1)
#
# Paths
load_k_shortest_paths(network=tntp_network, k=3, update_incidence_matrices=True)
# features_Z = []

# REad synthethic data which was generated under the assumption of path sets of size 2.
df = pd.read_csv(
    main_dir + '/input/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

# Generate synthetic node data
node_data = pd.DataFrame({'key': [node.key for node in tntp_network.nodes],
                          'income': np.random.rand(len(tntp_network.nodes)),
                          'population': np.random.rand(len(tntp_network.nodes))
                          })

## Exogenous features in utility function
_FEATURES_Z = ['tt_sd', 's']
# features_Z = []

# n_sparse_features = 0
# features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# # features_sparse = []
# _FEATURES_Z.extend(features_sparse)

# Prepare the training and validation dataset.
n_timepoints = len(df.timepoint.unique())
n_links = len(tntp_network.links)

# Add free flow travel times
df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_timepoints)

# TODO: exclude test data from transform to avoid data leakage
# df[features_Z + ['tt'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['traveltime'] + ['tt_ff']])

# # For splitting of training and validation set
# df['hour'] = 0
# df.loc[df['period'] <= 20, 'hour'] = 4
# df.loc[df['period'] >= 80, 'hour'] = 5
#
period_feature = 'hour'
df = df.rename(columns = {'period': 'hour'})

df = add_period_id(df, period_feature=period_feature)

period_keys = df[[period_feature, 'period_id']].drop_duplicates().reset_index().drop('index', axis=1).sort_values(period_feature)
print(period_keys)

X = get_design_tensor(Z=df[_FEATURES_Z + ['period_id']], n_links=n_links, n_timepoints=n_timepoints )
Y = get_design_tensor(y=df[['traveltime', 'counts']], n_links=n_links, n_timepoints=n_timepoints)

# df_T = df[df.hour.isin([0])]
df_T = df

XT = get_design_tensor(Z=df_T[_FEATURES_Z + ['period_id']], n_links=n_links, n_timepoints=len(df_T.timepoint.unique()))
YT = get_design_tensor(y=df_T[['traveltime', 'counts']], n_links=n_links, n_timepoints=len(df_T.timepoint.unique()))

# YT_train, YT_val = train_val_split_by_links(YT.numpy(), val_size=0.2)
# Y_train, Y_val = train_val_split_by_links(Y.numpy(), val_size=0.2)

Y_train, Y_val = train_val_split_by_links(Y.numpy(), val_size=0)
YT_train, YT_val = train_val_split_by_links(YT.numpy(), val_size=0)

_DTYPE = tf.float32

X_train, X_val = X, X
X_train, X_val, Y_train, Y_val = [tf.cast(i, _DTYPE) for i in [X_train, X_val, Y_train, Y_val]]

XT_train, XT_val = XT, XT
XT_train, XT_val, YT_train, YT_val = [tf.cast(i, _DTYPE) for i in [XT_train, XT_val, YT_train, YT_val]]

# Challenge is to estimate without knowing a historic OD but only generated trips.
# Thus, no deviation respect to od is added

# _LOSS_WEIGHTS = {'od': 0, 'theta': 0, 'traveltime': 1, 'flow': 1, 'equilibrium': 1,
#                  'ntrips': 0, 'prop_od': 0}
_LOSS_WEIGHTS = {'od': 0, 'theta': 0, 'traveltime': 1, 'flow': 1, 'equilibrium': 1, 'ntrips': 0, 'prop_od': 0} # For MLP


_BATCH_SIZE = None
_EQUILIBRIUM_STAGE = True
_ALTERNATING_OPTIMIZATION = False
_LR = {'learning': 1e-1, 'equilibrium': 1e-1}
_EPOCHS = {'learning': 20, 'equilibrium': 10}
# _BATCH_SIZE = None
# _EPOCHS = {'learning': 2, 'equilibrium': 1}
_EPOCHS_PRINT_INTERVAL = {'learning': 1, 'equilibrium': 1}
_XTICKS_SPACING = 5

# _BATCH_SIZE = 1
# _EQUILIBRIUM_STAGE = False
# _ALTERNATING_OPTIMIZATION = True
# _LR = {'learning': 1e-1, 'equilibrium': 1e-1}
# _EPOCHS = {'learning': 50, 'equilibrium': 1}
# # _EPOCHS_PRINT_INTERVAL = {'learning':10, 'equilibrium':_EPOCHS['equilibrium']}
# _EPOCHS_PRINT_INTERVAL = {'learning': 1, 'equilibrium': 1}
# _XTICKS_SPACING = 1

# _EQUILIBRIUM_STAGE = False
# _ALTERNATING_OPTIMIZATION = False
# _LR = {'learning': 1e-1, 'equilibrium': 1e-1}
# _EPOCHS = {'learning': 1000, 'equilibrium': 0}
# # _EPOCHS = {'learning': 20, 'equilibrium': 0}
# _EPOCHS_PRINT_INTERVAL = {'learning':10, 'equilibrium':_EPOCHS['equilibrium']}
# _XTICKS_SPACING = 50

# # For convergence using non-diagonal kernel matrix
# _LOSS_WEIGHTS = {'od': 0, 'theta': 0, 'traveltime': 1, 'flow': 1, 'equilibrium': 1,
#                  'ntrips': 0, 'prop_od': 0}

# With CPU
# _OPTIMIZERS = {'learning': tf.keras.optimizers.Adam(learning_rate=_LR['learning']),
#               'equilibrium': tf.keras.optimizers.Adam(learning_rate=_LR['equilibrium'])
#               }

# With GPU
_OPTIMIZERS = {'learning': tf.keras.optimizers.legacy.Adam(learning_rate=_LR['learning']),
               'equilibrium': tf.keras.optimizers.legacy.Adam(learning_rate=_LR['equilibrium'])
               }

_RELATIVE_GAP = 1e-3

_MOMENTUM_EQUILIBRIUM = 1

_LOSS_METRIC = z2score
# _LOSS_METRIC = zscore
# _LOSS_METRIC = nrmse
# _LOSS_METRIC = mse


# Models
list_models = ['equilibrium', 'odlulpe', 'tvodlulpe', 'tvodlulpe-kfold', 'tvodlulpe-outofsample']

# run_model = dict.fromkeys(list_models,True)
run_model = dict.fromkeys(list_models, False)

# run_model['equilibrium'] = True
# run_model['odlulpe'] = True
run_model['tvodlulpe'] = True
# run_model['tvodlulpe-kfold'] = True
# run_model['tvodlulpe-outofsample'] = True

models = {}

train_results_dfs = {}
val_results_dfs = {}

def create_mlp(network, dtype =_DTYPE):
    return MLP(n_links=len(network.links),
               free_flow_traveltimes=[link.bpr.tf for link in network.links],
               capacities=[link.bpr.k for link in network.links],
               kernel_constraint=KernelConstraint(
                   link_keys=[(link.key[0], link.key[1]) for link in network.links],
                   # adjacency_constraint=True,
                   dtype=dtype,
                   capacities=[link.bpr.k for link in network.links],
                   free_flow_traveltimes=[link.bpr.tf for link in network.links],
                   # diagonal=True,
                   # homogenous=True
               ),
               trainable = True,
               polynomial_layer= PolynomialLayer(poly_order=8,
                                                 trainable = True,
                                                 pretrain_weights=True,
                                                 alpha_prior = 1, beta_prior=2,
                                                 # kernel_constraint=tf.keras.constraints.NonNeg(),
                                                 ),
               alpha_relu=1,
               depth=1,
               dtype=dtype)

def create_bpr(network, dtype =_DTYPE):
    return BPR(keys=['alpha', 'beta'],
               # initial_values={'alpha': 0.15, 'beta': 4},
               # initial_values={'alpha': 1, 'beta': 1}, # Consistent with MLP initialization
               initial_values={'alpha': 0.15 * tf.ones(len(network.links), dtype = dtype),
                               'beta': 4 * tf.ones(len(network.links), dtype = dtype)},
               true_values={'alpha': 0.15, 'beta': 4},
               trainables={'alpha': True, 'beta':True},
               capacities = [link.bpr.k for link in network.links],
               free_flow_traveltimes =[link.bpr.tf for link in network.links],
               dtype = dtype
               )

def create_tvodlulpe_model_siouxfalls(network, dtype=_DTYPE, n_periods=1, features_Z=_FEATURES_Z, historic_g=None,
                                      performance_function=None):
    # optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           # trainables={'psc_factor': False, 'fixed_effect': False
                                           #     , 'traveltime': True, 'tt_sd': True, 's': True},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           time_varying=True,
                                           dtype=dtype
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])
    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])

    if performance_function is None:
        # performance_function = create_bpr(network = network, dtype = dtype)
        performance_function = create_mlp(network = network, dtype = dtype)

    od_parameters = ODParameters(key='od',
                                 initial_values=network.q.flatten(),
                                 true_values=network.q.flatten(),
                                 # historic_values={0: network.q.flatten()},
                                 # total_trips={0: np.sum(network.Q)},
                                 ods=network.ods,
                                 n_periods=n_periods,
                                 time_varying=True,
                                 trainable=True)

    generation_parameters = GenerationParameters(
        features_Z=['income', 'population'],
        initial_values={
            # 'income': 0,
            'fixed_effect': historic_g,
        },
        keys=['fixed_effect_od', 'fixed_effect_origin', 'fixed_effect_destination'],
        # true_values={'income': 0, 'fixed_effect': np.zeros_like(network.links)},
        # signs = {'income': '+','population': '+'},
        trainables={
            'fixed_effect': False, 'income': False, 'population': False,
            # 'fixed_effect_origin': False, 'fixed_effect_destination': False, 'fixed_effect_od': True
            'fixed_effect_origin': False, 'fixed_effect_destination': True, 'fixed_effect_od': False
        },
        pretrain_generation_weights=False,
        historic_g=historic_g,
        dtype=dtype
    )

    model = NESUELOGIT(
        key='tvodlulpe',
        network=network,
        dtype=dtype,
        utility=utility_parameters,
        performance_function=performance_function,
        generation=generation_parameters,
        od=od_parameters,
        n_periods=n_periods
    )

    return model, {'utility_parameters': utility_parameters, 'generation_parameters': generation_parameters,
                   'od_parameters': od_parameters, 'performance_function': performance_function}


if run_model['equilibrium']:
    print('Gradient based SUELOGIT')

    # To report runtime
    t0 = time.time()

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generated_trips = compute_generated_trips(q=isl.networks.denseQ(Q_historic).flatten()[np.newaxis, :],
                                              ods=tntp_network.ods)

    suelogit, _ = create_tvodlulpe_model_siouxfalls(network=tntp_network,
                                                    n_periods=n_periods,
                                                    historic_g=generated_trips)

    train_results_dfs['suelogit'], val_results_dfs['suelogit'] = suelogit.compute_equilibrium(
        XT_train,
        node_data=node_data,
        loss_metric=_LOSS_METRIC,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-1),
        batch_size=1,
        loss_weights={'od': 0, 'theta': 0, 'traveltime': 0, 'flow': 0, 'equilibrium': 1},
        threshold_relative_gap=1e-5,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=100)

    train_results_estimates, train_results_losses = suelogit.split_results(results=train_results_dfs['suelogit'])
    val_results_estimates, val_results_losses = suelogit.split_results(results=val_results_dfs['suelogit'])

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=val_results_dfs['suelogit'],
                                xticks_spacing=5, curves=['equilibrium'])

    fig, ax = plot_convergence_estimates(
        estimates=train_results_losses.assign(
            relative_gap=np.abs(train_results_losses['relative_gap']))[['epoch', 'relative_gap']],
        xticks_spacing=5)

    ax.set_yscale('log')
    ax.set_ylabel('relative gap (log scale)')

    plt.show()

    print(f'\nruntime: {time.time() - t0:0.1f} [s]')

if run_model['odlulpe']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix')

    # To report runtime
    t0 = time.time()

    performance_function = BPR(keys=['alpha', 'beta'],
                               initial_values={'alpha': 0.15, 'beta': 4},
                               # initial_values={'alpha': 0.15 * tf.ones(len(tntp_network.links), dtype = _DTYPE),
                               #                 'beta': 4 * tf.ones(len(tntp_network.links), dtype = _DTYPE)},
                               true_values={'alpha': 0.15, 'beta': 4},
                               trainables={'alpha': True, 'beta': True},
                               # trainables={'alpha': False, 'beta': False},
                               capacities=[link.bpr.k for link in tntp_network.links],
                               free_flow_traveltimes=[link.bpr.tf for link in tntp_network.links],
                               dtype=_DTYPE
                               )

    # # Estimation of linear function (MLP with a single parameter for the diagonal)
    # performance_function = BPR(keys=['alpha', 'beta'],
    #                            initial_values={'alpha': 1e-4, 'beta': 1},
    #                            true_values={'alpha': 0.15, 'beta': 4},
    #                            trainables={'alpha': True, 'beta':False},
    #                            # trainables={'alpha': False, 'beta': False},
    #                            capacities = [link.bpr.k for link in tntp_network.links],
    #                            free_flow_traveltimes =[link.bpr.tf for link in tntp_network.links],
    #                            dtype = _DTYPE
    #                            )

    # performance_function = MLP(n_links=len(tntp_network.links),
    #                            free_flow_traveltimes= [link.bpr.tf for link in tntp_network.links],
    #                            capacities=[link.bpr.k for link in tntp_network.links],
    #                            kernel_constraint = KernelConstraint(
    #                                capacities= [link.bpr.k for link in tntp_network.links],
    #                                free_flow_traveltimes= [link.bpr.tf for link in tntp_network.links],
    #                                diagonal = True,
    #                                # homogenous= True,
    #                            ),
    #                            dtype=_DTYPE)

    od_parameters = ODParameters(key='od',
                                 initial_values=isl.networks.denseQ(Q).flatten(),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 ods=tntp_network.ods,
                                 trainable=False)

    utility_parameters = UtilityParameters(features_Y=['traveltime'],
                                           features_Z=_FEATURES_Z,
                                           initial_values={'traveltime': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'traveltime': -1, 'tt_sd': -1.3, 's': -3},
                                           # signs={'traveltime': '-', 'tt_sd': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'traveltime': True, 'tt_sd': True, 's': True},
                                           dtype=_DTYPE
                                           )

    # utility_parameters.random_initializer((0,0),['traveltime','tt_sd','s'])
    # utility_parameters.initial_values = dict(utility_parameters.initial_values, traveltime = -1, c = -1, s = -1)

    generation_parameters = GenerationParameters(
        features_Z=['income', 'population'],
        initial_values={'income': 0,
                        'fixed_effect': od_parameters.compute_generated_trips(),
                        },
        keys=['fixed_effect_od', 'fixed_effect_origin', 'fixed_effect_destination'],
        # true_values={'income': 0, 'fixed_effect': np.zeros_like(tntp_network.links)},
        # signs = {'income': '+','population': '+'},
        trainables={
            'fixed_effect': True, 'income': False, 'population': False,
            # 'fixed_effect': True, 'income': False, 'population': False,
            'fixed_effect_origin': False, 'fixed_effect_destination': False, 'fixed_effect_od': True
            # 'fixed_effect_origin': False, 'fixed_effect_destination': True, 'fixed_effect_od': False
        },
        historic_g=od_parameters.compute_generated_trips(),
        dtype=_DTYPE
    )

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    models['odlulpe'] = NESUELOGIT(
        key='odlulpe',
        network=tntp_network,
        dtype=_DTYPE,
        equilibrator=equilibrator,
        utility=utility_parameters,
        generation=generation_parameters,
        performance_function=performance_function,
        od=od_parameters,
    )

    train_results_dfs['odlulpe'], val_results_dfs['odlulpe'] = models['odlulpe'].fit(
        X_train, Y_train, X_val, Y_val,
        optimizers=_OPTIMIZERS,
        node_data=node_data,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights=dict(_LOSS_WEIGHTS),
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        alternating_optimization=_ALTERNATING_OPTIMIZATION,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    train_results_estimates, train_results_losses = models['odlulpe'].split_results(
        results=train_results_dfs['odlulpe'])
    val_results_estimates, val_results_losses = models['odlulpe'].split_results(results=val_results_dfs['odlulpe'])

    if _PLOTS:
        sns.scatterplot(data=pd.DataFrame({'link flow': models['odlulpe'].predict_flow().numpy().flatten(),
                                           'travel time': models['odlulpe'].predict_traveltime().numpy().flatten(),
                                           # 'capacity': [link.bpr.k for link in odlulpe.network.links]
                                           }),
                        x='link flow', y='travel time')

        plot_predictive_performance(train_losses=train_results_losses, val_losses=val_results_dfs['odlulpe'],
                                    show_validation=True, xticks_spacing=_XTICKS_SPACING)

        plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    epochs_end_learning_stage=_EPOCHS['learning'])

        plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
                                    yaxis_label='mape')

        plot_convergence_estimates(estimates=train_results_estimates. \
                                   assign(rr=train_results_estimates['tt_sd'] / train_results_estimates['traveltime'])[
            ['epoch', 'rr']],
                                   # true_values={'rr':odlulpe.utility.true_values['tt_sd']/odlulpe.utility.true_values['traveltime']},
                                   xticks_spacing=_XTICKS_SPACING)

        if models['odlulpe'].performance_function.type == 'BPR':
            plot_convergence_estimates(estimates=train_results_estimates[['epoch', 'alpha', 'beta']],
                                       true_values=models['odlulpe'].performance_function.true_values,
                                       xticks_spacing=_XTICKS_SPACING)

        plt.show()

        Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic,
              'estimated': tf.sparse.to_dense(models['odlulpe'].Q).numpy()}

        plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3),
                             figsize=(12, 4))

        fig, ax = plot_convergence_estimates(
            estimates=train_results_losses.assign(
                relative_gap=np.abs(train_results_losses['relative_gap']))[['epoch', 'relative_gap']],
            xticks_spacing=_XTICKS_SPACING)

        ax.set_yscale('log')
        ax.set_ylabel('relative gap (log scale)')

        plt.show()

    print(
        f"\ntheta = {dict(zip(utility_parameters.true_values.keys(), list(np.round(models['odlulpe'].theta.numpy().flatten(), 2))))}, "
        f"rr = {train_results_estimates.eval('tt_sd/traveltime').values[-1]:0.2f}")

    if models['odlulpe'].performance_function.type == 'BPR':
        print(f"alpha = {np.mean(models['odlulpe'].alpha): 0.2f}, beta  = {np.mean(models['odlulpe'].beta): 0.2f}")

    print(
        f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(models['odlulpe'].q - tntp_network.q.flatten())): 0.2f}")

    print(compute_insample_outofsample_error(Y=Y_train,
                                             true_counts=df.true_counts.values[0:tntp_network.get_n_links()],
                                             true_traveltimes=df.true_traveltime.values[0:tntp_network.get_n_links()],
                                             model=models['odlulpe'],
                                             metric=mse))

    metrics_df = models['odlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                        X=X_train, Y=Y_train).assign(dataset='training')
    metrics_df = pd.concat([metrics_df,
                            models['odlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                                   X=X_val, Y=Y_val).assign(dataset='validation')])

    with pd.option_context('display.float_format', '{:0.2g}'.format):
        print(pd.pivot(metrics_df, index=['component', 'dataset'], columns=['metric'])['value'])

    print(f'\nruntime: {time.time() - t0:0.1f} [s]')

if run_model['tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generation_factors = compute_generation_factors(period_column=XT[:, :, -1, None].numpy(),
                                                    flow_column=YT[:, :, 1, None].numpy(), reference_period=0)

    generated_trips = compute_generated_trips(q=isl.networks.denseQ(Q_historic).flatten()[np.newaxis, :],
                                              ods=tntp_network.ods)

    models['tvodlulpe'], _ = create_tvodlulpe_model_siouxfalls(
        n_periods=n_periods, network=tntp_network,
        historic_g= generation_factors.values[:,np.newaxis]*generated_trips
        # historic_g=generated_trips
    )

    train_results_dfs['tvodlulpe'], val_results_dfs['tvodlulpe'] = models['tvodlulpe'].fit(
        XT_train, YT_train, XT_val, YT_val,
        # X_train, Y_train, X_val, Y_val,
        # X_train, Y_train, X_val, Y_val,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        node_data=node_data,
        batch_size=_BATCH_SIZE,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    # Y_pred = models['tvodlulpe'].predict(XT_train,
    #                             period_dict={k: v for k, v in models['tvodlulpe'].period_dict.items()},
    #                             node_data=node_data,
    #                             loss_metric=_LOSS_METRIC,
    #                             optimizer=_OPTIMIZERS['equilibrium'],
    #                             # batch_size= 1,
    #                             loss_weights={'equilibrium': 1},
    #                             threshold_relative_gap=1e-3,  # _RELATIVE_GAP,
    #                             epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
    #                             epochs=0
    #                             )

    # MLP weights
    mlp_weights = models['tvodlulpe'].performance_function.weights

    link_flow_interaction_matrix = mlp_weights[1].numpy()

    matrix_df = pd.DataFrame({'link_1': pd.Series([], dtype=int),
                              'link_2': pd.Series([], dtype=int),
                              'weight': pd.Series([], dtype=float)})

    # rows, cols = link_flow_interaction_matrix.shape
    #
    # counter = 0
    # for link_1 in range(0, rows):
    #     for link_2 in range(0, cols):
    #         matrix_df.loc[counter] = [int(link_1 + 1), int(link_2 + 1), Q[(origin, destination)]]
    #         counter += 1
    #
    # od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')
    #
    # sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues", vmin=vmin, vmax=vmax, ax=axi)

    if _PLOTS:

        # Plots travel time against flow

        plot_flow_vs_traveltime(model = models['tvodlulpe'],
                                observed_traveltime=Y[:, :, 0],
                                observed_flow= Y[:,:,1]
                                )
        # plt.show()

        plot_mlp_performance_functions(model = models['tvodlulpe'],
                                       network = tntp_network,
                                       marginal = False,
                                       alpha = 1,
                                       beta = 2
                                       # selected_links = np.random.choice(range(tntp_network.get_n_links()), 10, replace=False)
                                       )

        # plt.show()

        selected_links = np.random.choice(range(tntp_network.get_n_links()), 10, replace=False)

        # Plot with bpr used the priors of the BPR parameters used to pretrain the MLP
        plot_mlp_performance_functions(model=models['tvodlulpe'],
                                       network=tntp_network,
                                       marginal=False,
                                       alpha=1,
                                       beta=2,
                                       selected_links = selected_links,
                                       palette = sns.color_palette("hls", len(selected_links))
                                       )

        plt.show()


        plot_mlp_performance_functions(model=models['tvodlulpe'],
                                       network=tntp_network,
                                       marginal=True,
                                       alpha=1,
                                       beta=2,
                                       selected_links = selected_links
                                       )

        plt.show()

        # # This shows that the model is able to learn the true BPR function with a polynonomial kernel
        # plot_performance_functions(model=models['tvodlulpe'],
        #                            network=tntp_network,
        #                            marginal=True,
        #                            alpha=0.15,
        #                            beta=4,
        #                            selected_links = selected_links
        #                            )
        #
        # plt.show()

        # Plot heatmap with flows of top od pairs
        plot_top_od_flows_periods(models['tvodlulpe'], period_keys=period_keys, period_feature=period_feature, top_k=20,
                                  historic_od=tntp_network.q.flatten())

        plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'],
                                    val_losses=val_results_dfs['tvodlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    epochs_end_learning_stage=_EPOCHS['learning'])

        plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'],
                                    val_losses=val_results_dfs['tvodlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
                                    yaxis_label='mape')

        fig, ax = plot_convergence_estimates(
            estimates=train_results_dfs['tvodlulpe'].assign(
                relative_gap=np.abs(train_results_dfs['tvodlulpe']['relative_gap']))[['epoch', 'relative_gap']],
            xticks_spacing=_XTICKS_SPACING)

        ax.set_yscale('log')
        ax.set_ylabel('relative gap (log scale)')

        plt.show()

        if models['tvodlulpe'].performance_function.type == 'bpr':
            plot_convergence_estimates(estimates=train_results_dfs['tvodlulpe'][['epoch', 'alpha', 'beta']],
                                       xticks_spacing=_XTICKS_SPACING,
                                       true_values=models['tvodlulpe'].performance_function.parameters.true_values,
                                       )

            # sns.displot(pd.melt(pd.DataFrame({'alpha': [float(tvodlulpe.performance_function._alpha)],
            #                                   'beta': [float(tvodlulpe.performance_function.beta)]}), var_name='parameters'),
            #             x="value", hue="parameters", multiple="stack", kind="kde", alpha=0.8)

            plt.show()

        # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
        theta_df = plot_utility_parameters_periods(models['tvodlulpe'], period_keys=period_keys, period_feature=period_feature)

        plt.show()

        rr_df = theta_df.apply(compute_rr, axis=1).reset_index().rename(columns={'index': period_feature, 0: 'rr'})

        sns.lineplot(data=rr_df, x=period_feature, y="rr")

        plt.show()

        sns.displot(pd.DataFrame({'fixed_effect': np.array(models['tvodlulpe'].fixed_effect)}),
                    x="fixed_effect", multiple="stack", kind="kde", alpha=0.8)

        plt.show()

        Qs = {'true': tntp_network.OD.Q_true,
              # 'historic': Q_historic,
              'estimated': tf.sparse.to_dense(models['tvodlulpe'].Q).numpy()}

        plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, len(Qs.keys())),
                             figsize=(4*len(Qs.keys()), 4))

        plt.show()

    print(f"theta = "
          f"{dict(zip(models['tvodlulpe'].utility.true_values.keys(), list(np.mean(models['tvodlulpe'].theta.numpy(), axis=0))))}")
    print(f"kappa= "
          f"{dict(zip(models['tvodlulpe'].generation.features, list(np.mean(models['tvodlulpe'].kappa.numpy(), axis=0))))}")

    if models['tvodlulpe'].performance_function.type == 'bpr':
        print(f"alpha = {np.mean(models['tvodlulpe'].performance_function.alpha): 0.2f}, "
              f"beta  = {np.mean(models['tvodlulpe'].performance_function.beta): 0.2f}")

    print(f"Avg abs diff of observed and estimated OD: "
          f"{np.mean(np.abs(models['tvodlulpe'].q - tntp_network.q.flatten())): 0.2f}")

    print(f"Avg observed OD: {np.mean(np.abs(tntp_network.q.flatten())): 0.2f}")

    metrics_df = models['tvodlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                          X=XT_train, Y=YT_train).assign(dataset='training')

    # metrics_df = pd.concat([metrics_df,
    #                      compute_benchmark_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
    #                                                Y_ref=Y_train, Y=Y_train).assign(dataset='training', stage='benchmark')])
    metrics_df = pd.concat([metrics_df,
                            models['tvodlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                                     X=XT_val, Y=YT_val).assign(dataset='validation'),
                            compute_benchmark_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2}, Y_ref=YT_train,
                                                      Y=YT_val).assign(
                                dataset='benchmark')
                            ])

    with pd.option_context('display.float_format', '{:0.2g}'.format):
        print(pd.pivot(metrics_df, index=['component', 'dataset'], columns=['metric'])['value'])


    # # Benchmark
    # for i in range(Y_pred.shape[-1]):
    #     print(mape(actual = [:,:,i,None],
    #          predicted=(np.nanmean(YT_train.numpy().reshape(-1,YT_train.shape[-1]),0)*np.ones_like(Y_pred))[:,:,i,None]))

    # Model
    # for i in range(Y_pred.shape[-1]):
    #     print(mape(actual = YT_val[:,:,i,None], predicted=Y_pred[:,:,i,None]))

if run_model['tvodlulpe-kfold']:
    print('\ntvodlulpe-kfold: Time specific utility and OD, link performance parameters, no historic OD')

    n_periods = len(np.unique(X[:, :, -1].numpy().flatten()))

    generated_trips = compute_generated_trips(q=isl.networks.denseQ(Q_historic).flatten()[np.newaxis, :],
                                              ods=tntp_network.ods)

    model, _ = create_tvodlulpe_model_siouxfalls(
        n_periods=n_periods, network=tntp_network, historic_g=generated_trips)

    metrics_kfold_df = train_kfold(
        n_splits=2,
        random_state=_SEED,
        model=model,
        X=X, Y=Y,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        node_data=node_data,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        # batch_size=1,
        batch_size=_BATCH_SIZE,
        # epochs={'learning': 3, 'equilibrium': 5},
        epochs= _EPOCHS,
        # epochs_print_interval={'learning': 100, 'equilibrium': 100},

    )

    metrics_kfold_df.to_csv(f"./output/experiments/{datetime.now().strftime('%y%m%d%H%M%S')}_kfold_{network_name}.csv")

    # TODO: Add coefficient of variation and save experiments results, compute percentage reduction between final and initial
    with pd.option_context('display.float_format', '{:0.2g}'.format):
        # print(metrics_kfold_df[metrics_kfold_df.component.isin(['flow','traveltime'])].\
        #       groupby(['dataset', 'component', 'metric', 'stage'])['value'].\
        #       aggregate(['mean', 'std']))
        print(metrics_kfold_df. \
              groupby(['dataset', 'component', 'metric', 'stage'])['value']. \
              aggregate(['mean', 'std']))

    # metrics_kfold_df = pd.read_csv(f"./output/experiments/{230525002028}_kfold_{network_name}.csv")
    # metrics_kfold_df = pd.read_csv(f"./output/experiments/{230525213615}_kfold_{network_name}.csv")

    plot_metrics_kfold(df=metrics_kfold_df[metrics_kfold_df.component.isin(['flow', 'traveltime'])])

    plt.show()

if run_model['tvodlulpe-outofsample']:
    print('\ntvodlulpe-outofsample')

    # def save_model(reference_model, X):
    #
    #     X = reference_model.set_period_ids(X.numpy())
    #
    #     reference_model(X)
    #     reference_model.predict(X=X*10)
    #     reference_model.save('output/models/test.tf')
    #
    #     new_model = tf.keras.models.load_model('output/models/test.tf', compile = False)
    #     new_model.weights
    #     new_model.predict(X * 10)
    #     new_model.summary()
    #
    #     # new_model.compile()

    # generation_factors = (df_T.groupby('period_id')['counts'].mean() / df_T[df_T.period_id == 2].counts.mean()).values

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generation_factors = compute_generation_factors(period_column=XT[:, :, -1, None].numpy(),
                                                    flow_column=YT[:, :, 1, None].numpy(), reference_period=2)

    generated_trips = compute_generated_trips(q=isl.networks.denseQ(Q_historic).flatten()[np.newaxis, :],
                                              ods=tntp_network.ods)

    reference_model, _ = create_tvodlulpe_model_siouxfalls(
        n_periods=n_periods, network=tntp_network,
        historic_g=generation_factors.values[:, np.newaxis] * generated_trips)

    reference_model.fit(
        XT_train, YT_train,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        node_data=node_data,
        # batch_size=_BATCH_SIZE,
        batch_size=None,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        # epochs=_EPOCHS
        epochs={'learning': 10, 'equilibrium': 5}
    )

    print(reference_model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2}, X=XT_train, Y=YT_train))

    model = create_inference_model(creation_method=create_tvodlulpe_model_siouxfalls, reference_model=reference_model)

    # model.summary()

    model.predict(XT_val,
                  period_dict={k: v for k, v in reference_model.period_dict.items()},
                  node_data=node_data,
                  loss_metric=_LOSS_METRIC,
                  optimizer=_OPTIMIZERS['equilibrium'],
                  # batch_size= 1,
                  loss_weights={'equilibrium': 1},
                  threshold_relative_gap=1e-3,  # _RELATIVE_GAP,
                  epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
                  epochs=4)

    print(model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2}, X=XT_val, Y=YT_val))

    # Bad model because it has not been trained
    other_model = create_tvodlulpe_model_siouxfalls(
        historic_g=generated_trips * generation_factors.values[:, np.newaxis], network=tntp_network)[0]

    other_model.build()
    other_model.setup_period_ids(X_train=XT_val, node_data=node_data)
    # other_model.forward(XT_val)

    print(other_model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2}, X=XT_val, Y=YT_val))

    # TODO: Build a beenchmark model here

sys.exit()

# Write predictions
predictions = pd.DataFrame({'link_key': list(tntp_network.links_keys) * Y_train.shape[0],
                            'observed_traveltime': Y_train[:, :, 0].numpy().flatten(),
                            'observed_flow': Y_train[:, :, 1].numpy().flatten()})

predictions['period'] = df.period

# TODO: predictions are not supported for tvodlulpe model yet

# for model in models.values():
#
#     # model = odlulptte
#
#     predicted_flows = model.predicted_flow()
#     predicted_traveltimes = model.traveltimes()
#
#     predictions['predicted_traveltime_' + model.key] = np.tile(predicted_traveltimes, (Y_train.shape[0], 1)).flatten()
#     predictions['predicted_flow_' + model.key] = np.tile(predicted_flows, (Y_train.shape[0], 1)).flatten()
#
# predictions.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_predictions_{network_name}.csv")


# Write csv file with estimation results
train_results_df, val_results_df \
    = map(lambda x: pd.concat([results.assign(model=model)[['model'] + list(results.columns)]
                               for model, results in x.items()], axis=0), [train_results_dfs, val_results_dfs])

train_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_results_{network_name}.csv")
val_results_df.to_csv(
    f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_validation_results_{network_name}.csv")

# Summary of model parameters
results = pd.DataFrame({'parameter': [], 'model': []})

for model in models.values():
    model_results = {**dict(zip(['traveltime'] + _FEATURES_Z, list(np.mean(model.theta.numpy(), axis=0)))),
                     **{'rr': float(model.get_parameters_estimates().eval('tt_sd/traveltime').iloc[0]),
                        'fixed_effect_mean': np.mean(model.fixed_effect),
                        'fixed_effect_std': np.std(model.fixed_effect),
                        'od_mean': np.mean(model.q),
                        'od_std': np.std(model.q)
                        }}

    if model.performance_function.type == 'bpr':
        model_results = {**model_results, **{'alpha_mean': np.mean(model.performance_function.alpha),
                                             'alpha_std': np.std(model.performance_function.alpha),
                                             'beta_mean': np.mean(model.performance_function.beta),
                                             'beta_std': np.std(model.performance_function.beta)}}

    model_results = pd.DataFrame({'parameter': model_results.keys(), 'values': model_results.values()}). \
        assign(model=model.key)

    results = pd.concat([results, model_results])

print(results.pivot_table(index=['parameter'], columns='model', values='values', sort=False).round(4))

## Summary of models goodness of fit

results_losses = pd.DataFrame({})
loss_columns = ['loss_flow', 'loss_traveltime', 'loss_equilibrium', 'loss_total']

for i, model in models.items():
    results_losses_model = model.split_results(train_results_dfs[model.key])[1].assign(model=model.key)
    results_losses_model = results_losses_model[results_losses_model.epoch == _EPOCHS['learning']].iloc[[0]]
    results_losses = pd.concat([results_losses, results_losses_model])

results_losses[loss_columns] = (results_losses[loss_columns] - 1) * 100

print(results_losses[['model'] + loss_columns].round(1))

## Plot of convergence toward true reliabiloty ratio (rr) across models

train_estimates = {}
train_losses = {}

for i, model in models.items():
    train_estimates[model.key], train_losses[model.key] = model.split_results(results=train_results_dfs[model.key])

    train_estimates[model.key]['model'] = model.key

train_estimates_df = pd.concat(train_estimates.values())

train_estimates_df['rr'] = train_estimates_df['tt_sd'] / train_estimates_df['traveltime']

estimates = train_estimates_df[['epoch', 'model', 'rr']].reset_index().drop('index', axis=1)
estimates = estimates[estimates.epoch != 0]

fig, ax = plt.subplots(nrows=1, ncols=1)

g = sns.lineplot(data=estimates, x='epoch', hue='model', y='rr')

ax.hlines(y=compute_rr(utility_parameters.true_values), xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(),
          linestyle='--', label='truth')

ax.set_ylabel('reliability ratio')

plt.ylim(ymin=0)

plt.show()

'''
'''

import os
import sys
from pathlib import Path
# from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
plt.ion()
_PLOTS = False

import seaborn as sns
from datetime import datetime

import numpy as np
import pandas as pd
import random
# import cudf as pd
import tensorflow as tf
import isuelogit as isl
import glob
import time

# To run code on CPU (two times much faster than GPU)
tf.config.set_visible_devices([], 'GPU')

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)
isl.config.dirs['read_network_data'] = "input/network-data/fresno/"

# Internal modules

from pesuelogit.visualizations import plot_convergence_estimates
from pesuelogit.networks import load_k_shortest_paths, read_paths, build_fresno_network, \
    Equilibrator, sparsify_OD, ColumnGenerator, read_OD
from pesuelogit.etl import get_design_tensor, get_y_tensor, data_curation, temporal_split, add_period_id, get_tensors_by_year

# Internal modules
from nesuelogit.models import NESUELOGIT, ODParameters, UtilityParameters, BPR, MLP, KernelConstraint, \
    GenerationParameters, train_val_split_by_links, train_kfold, compute_generated_trips, compute_generation_factors, \
    create_inference_model, PolynomialLayer
from nesuelogit.visualizations import  plot_predictive_performance, plot_metrics_kfold, plot_top_od_flows_periods, \
    plot_utility_parameters_periods, plot_rr_by_period, plot_rr_by_period_models, plot_total_trips_models
from nesuelogit.metrics import mse, btcg_mse, mnrmse, mape, nrmse, r2, zscore

# Seed for reproducibility
_SEED = 2023
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# To write estimation report and set seed for all algorithms involving some randomness
# estimation_reporter = isl.writer.Reporter(
#     folderpath=isl.config.dirs['output_folder'] + 'estimations/' + 'Fresno', seed=_SEED)

## Build Fresno network
fresno_network = build_fresno_network()

# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type != 'LWRLK']

## Read OD matrix
# TODO: option to specify path to read OD matrix
read_OD(network=fresno_network, sparse=True)

# np.sum(fresno_network.Q)

# Read paths
# read_paths(network=fresno_network, update_incidence_matrices=True, filename='paths-fresno.csv')
# read_paths(network=fresno_network, update_incidence_matrices=True, filename = 'paths-full-model-fresno.csv')

# For quick testing (do not need to read_paths before)
Q = fresno_network.load_OD(sparsify_OD(fresno_network.Q, prop_od_pairs=0.99))
load_k_shortest_paths(network=fresno_network, k=2, update_incidence_matrices=True)

## Read spatiotemporal data
folderpath = isl.config.dirs['read_network_data'] + 'links/spatiotemporal-data/'
df = pd.concat([pd.read_csv(file) for file in glob.glob(folderpath + "*link-data*")], axis=0)
df.hour.unique()

# TODO: Check why there are missing dates, e.g. October 1, 2019
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Select data from Tuesday to Thursday
df = df[df['date'].dt.dayofweek.between(1, 3)]
# df = df[df['date'].dt.dayofweek.between(1,1)]
# df = df[df['date'].dt.dayofweek.between(2,2)]
# df = df[df['date'].dt.year == 2019]
# df['date'].dt.dayofweek.unique()
# len(sorted(df['date']).unique())
# df['period'] = df.period.map(hash)

# Add id for period and respecting the temporal order
# periods_keys = dict(zip(sorted(df['period'].unique()), range(len(sorted(df['period'].unique())))))

# - By hour
period_feature = 'hour'

df['period'] = df['date'].astype(str) + '-' + df[period_feature].astype(str)

df = add_period_id(df, period_feature=period_feature)
period_keys = df[[period_feature,'period_id']].drop_duplicates().reset_index().drop('index',axis =1).sort_values('hour')
print(period_keys.reset_index().drop('index', axis = 1).T, '\n')

# - By hour

# df1 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2019-10-01-fresno-link-data.csv')
# df1['date'] = "2019-10-01"
# df1['period'] = 0
#
# df2 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2020-10-06-fresno-link-data.csv')
# df2['date'] = "2020-10-06"
# df2['period'] = 1
#
# df = pd.concat([df1, df2], axis=0)

## Data curation

# df['tt_ff'] = np.tile([link.bpr.tf for link in fresno_network.links],len(df.date.unique())*len(df.hour.unique()))
# df['tt_ff'] = df['tf_inrix']

df['tt_ff'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_ref_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_ref_avg == 0),'tt_ff'] = float('nan')

df['tt_avg'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_hist_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_hist_avg == 0),'tt_avg'] = float('nan')

tt_sd_adj = df.groupby(['period_id','link_key'])[['tt_avg']].std().reset_index().rename(columns = {'tt_avg': 'tt_sd_adj'})

df = df.merge(tt_sd_adj, on = ['period_id','link_key'])

df = data_curation(df)

df['tt_sd'] = df['tt_sd_adj']

# Units of travel time features are converted from hours to minutes
df['tt_sd'] = df['tt_sd']*60
df['tt_avg'] = df['tt_avg']*60
df['tt_ff'] = df['tt_ff']*60

# Nodes data

node_data = pd.read_csv(isl.config.dirs['read_network_data'] + 'nodes/fresno-nodes-gis-data.csv')
node_data['population'] = node_data['pop_tract']#/node_data['nodes_tract']
#TODO: Compute area of each census tract and normalize number of stops for that
# node_data['bus_stops'] = node_data['stops_tract']/node_data['pop_tract']
node_data['bus_stops'] = node_data['stops_tract']#/node_data['nodes_tract']
node_data['income'] = node_data['median_inc']

features_generation = ['population','income', 'bus_stops']

node_data = node_data[['key','type'] + features_generation]

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(node_data[features_generation])
node_data[features_generation] = imp_mean.transform(node_data[features_generation])

scaler = preprocessing.StandardScaler().fit(node_data[features_generation].values)
# scaler = preprocessing.MinMaxScaler().fit(node_data[features_generation].values)
node_data[features_generation] = scaler.transform(node_data[features_generation].values)

## Utility function

_FEATURES_Z = ['tt_sd', 'median_inc', 'incidents', 'bus_stops', 'intersections']
# _FEATURES_Z = ['tt_sd']
# _FEATURES_Z = []

# utility_parameters.constant_initializer(0)

## Data processing

n_links = len(fresno_network.links)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df.date.dt.year
df.hour.unique()

# Select only dates used for previous paper
# df = df.query('date == "2019-10-01"  | date == "2020-10-06"')
# df = df.query('date == "2019-10-01"')
# df = df.query('hour == 16')
# df = df.query('hour == 17')
# df = df.query('hour == 16 | hour == 17')
# df = df.query('hour == 17')

print(df.query('year == 2019')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

print(df.query('year == 2020')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

# Normalization of features to range [0,1]

# (TODO: may enable normalization in get_design_tensor method. See if tensorflow have it)
# TODO: exclude test data from transform to avoid data leakage
# df[_FEATURES_Z + ['tt_avg'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[_FEATURES_Z + ['tt_avg'] + ['tt_ff']])

# Set free flow travel times
# tt_ff_links = df.query('link_type == "LWRLK"').groupby('link_key')['tt_ff'].min()
tt_ff_links = df.groupby('link_key')['tt_ff'].min()
# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type == "LWRLK"]
for link in fresno_network.links:
    fresno_network.links_dict[link.key].performance_function.tf = float(tt_ff_links[tt_ff_links.index==str(link.key)].iloc[0])

tt_ff_links.mean()
df[['tt_avg','tt_ff','tf_inrix']].mean()

df['tt_ff'] = df.groupby('link_key')['tt_ff'].transform(lambda x: x.min())


# EDA

obs_date = df.groupby('date')['hour'].count()

df.groupby('date')[['speed_sd','speed_avg', 'counts']].mean().assign(total_obs = obs_date)

## Training and validation sets

_DTYPE = tf.float32

# Include only data between 4pm and 5pm
X, Y = get_tensors_by_year(df[df.hour == 16], features_Z = _FEATURES_Z, network = fresno_network)
# Include hourly data between 6AM and 8PM (15 hour intervals)
# XT, YT = get_tensors_by_year(df, features_Z = _FEATURES_Z)
# XT, YT = get_tensors_by_year(df[df.hour.isin(range(14,18))], features_Z = _FEATURES_Z, network = fresno_network)
XT, YT = get_tensors_by_year(df[df.hour.isin([6,7,8, 15,16,17])], features_Z = _FEATURES_Z, network = fresno_network)

# Split to comply with temporal ordering
# X_train, X_val, Y_train, Y_val = temporal_split(X[2019].numpy(), Y[2019].numpy(), n_days = X[2019].shape[0])


# X_train, X_val, Y_train, Y_val = X[2020], X[2019], Y[2020], Y[2019]
X_train, X_val, Y_train, Y_val = map(lambda x: tf.cast(x, dtype = _DTYPE), [X[2019], X[2020], Y[2019], Y[2020]])
XT_train, XT_val, YT_train, YT_val = map(lambda x: tf.cast(x, dtype = _DTYPE), [XT[2019], XT[2020], YT[2019], YT[2020]])

# # Remove validation set to reduce computation costs
# X_val, Y_val = None, None
# XT_val, YT_val = None, None

# # Split by links
# X_train, X_val = X_train, X_train
# Y_train, Y_val = train_val_split_by_links(Y_train.numpy(),val_size=0.2)
# X_train, X_val, Y_train, Y_val = [tf.cast(i, _DTYPE) for i in [X_train, X_val, Y_train, Y_val]]
#
# XT_train, XT_val = XT_train, XT_train
# YT_train, YT_val = train_val_split_by_links(YT_train.numpy(),val_size=0.2)
# XT_train, XT_val, YT_train, YT_val = [tf.cast(i, _DTYPE) for i in [XT_train, XT_val, YT_train, YT_val]]




_RELATIVE_GAP = 1e-2

_FIXED_EFFECT = False

# _LOSS_METRIC  = mnrmse
# _LOSS_METRIC  = nrmse
_LOSS_METRIC  = zscore

# _LOSS_WEIGHTS ={'od': 0, 'traveltime': 1, 'flow': 1, 'eq_flow': 3, 'prop_od': 0, 'ntrips': 0}
# _MOMENTUM_EQUILIBRIUM = 0.99
_MOMENTUM_EQUILIBRIUM = 1

# Including historic OD matrix
# _LOSS_WEIGHTS ={'od': 1, 'traveltime': 1, 'flow': 1, 'eq_flow': 1}
# _MOMENTUM_EQUILIBRIUM = 0.99

# _LOSS_METRIC = mse
# _LOSS_WEIGHTS ={'od': 1, 'theta': 0, 'traveltime': 1e10, 'flow': 1, 'eq_flow': 1}

#_LOSS_METRIC  = btcg_mse
#_LOSS_METRIC  = mnrmse


# print(f"Relative gap threshold: {_RELATIVE_GAP}, "
#       f"Learning rate: {_LR}, "
#       f"Batch size: {_BATCH_SIZE}")

_LOSS_WEIGHTS ={'od': 0, 'traveltime': 1, 'flow': 1, 'eq_flow': 4}
_EQUILIBRIUM_STAGE = True
_ALTERNATING_OPTIMIZATION = False

_LR = {'learning': 1e-1, 'equilibrium': 1e-1}

_BATCH_SIZE = 1 #16
# _EPOCHS = {'learning': 20, 'equilibrium': 10}
_EPOCHS = {'learning': 1, 'equilibrium': 1}
# _XTICKS_SPACING = 2
_XTICKS_SPACING = 5
_EPOCHS_PRINT_INTERVAL = {'learning':1, 'equilibrium':1}
# With data from a single day
# _EPOCHS = {'learning': 60, 'equilibrium': 30}


# _BATCH_SIZE = 16
# _EPOCHS = {'learning': 200, 'equilibrium': 100} # If TVODLULPE
# _EPOCHS_PRINT_INTERVAL = {'learning':1, 'equilibrium':1}
# _XTICKS_SPACING = 50

# _LOSS_WEIGHTS ={'od': 0, 'traveltime': 1, 'flow': 1, 'eq_flow': 1e-1}
# _EQUILIBRIUM_STAGE = False
# _ALTERNATING_OPTIMIZATION = True
# _LR = {'learning': 1e-1, 'equilibrium': 5e-1}
# _EPOCHS = {'learning': 100, 'equilibrium': 1}
# # _EPOCHS = {'learning': 20, 'equilibrium': 0}
# _EPOCHS_PRINT_INTERVAL = {'learning':1, 'equilibrium':2}
# _XTICKS_SPACING = 5

# _LOSS_WEIGHTS ={'od': 0, 'traveltime': 1, 'flow': 1, 'eq_flow': 4}
# _EQUILIBRIUM_STAGE = False
# _ALTERNATING_OPTIMIZATION = False
# _LR = {'learning': 1e-1, 'equilibrium': 1e-1}
# _EPOCHS = {'learning': 200, 'equilibrium': 4}
# # _EPOCHS = {'learning': 20, 'equilibrium': 0}
# _EPOCHS_PRINT_INTERVAL = {'learning':1, 'equilibrium':2}
# _XTICKS_SPACING = 5

# With CPU
# _OPTIMIZERS = {'learning': tf.keras.optimizers.Adam(learning_rate=_LR['learning']),
#               'equilibrium': tf.keras.optimizers.Adam(learning_rate=_LR['equilibrium'])
#               }

# With GPU
_OPTIMIZERS = {'learning': tf.keras.optimizers.legacy.Adam(learning_rate=_LR['learning']),
              'equilibrium': tf.keras.optimizers.legacy.Adam(learning_rate=_LR['equilibrium'])
              }

#Models
# run_model = dict.fromkeys(['equilibrium', 'lue', 'lpe', 'ode', 'odlue', 'odlulpe-1','odlulpe', 'tvodlulpe'], True)
run_model = dict.fromkeys(['equilibrium','odlulpe', 'tvodlulpe', 'tvodlulpe-kfold', 'tvodlulpe-outofsample'], False)

# run_model.update(dict.fromkeys(['lue', 'odlue', 'odlulpe'], True))
# run_model = dict.fromkeys( for i in ['lue', 'odlue', 'odlulpe'], True)
# run_model['equilibrium'] = True
# run_model['lue'] = True
# run_model['lpe'] = True
# run_model['ode'] = True
# run_model['odlue'] = True
# run_model['val_tvodlulpe'] = True

# run_model['odlulpe'] = True
if run_model['odlulpe']:
    _EPOCHS = {k:int(v*2) for k,v in _EPOCHS.items()}

# _EPOCHS = {k:int(v*90/16) for k,v in _EPOCHS.items()}

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
                               adjacency_constraint=True,
                               dtype=dtype,
                               capacities=[link.bpr.k for link in network.links],
                               free_flow_traveltimes=[link.bpr.tf for link in network.links],
                               diagonal=True,
                               symmetric=True,
                           ),
                           trainable = True,
                           polynomial_layer= PolynomialLayer(poly_order=4, trainable = True, pretrain_weights=True),
                           depth=1,
                           dtype=dtype)

def create_bpr(network, dtype =_DTYPE):
    return BPR(keys=['alpha', 'beta'],
               # initial_values={'alpha': 1, 'beta': 1}, # Consistent with MLP initialization
               initial_values={'alpha': 0.15 * tf.ones(len(network.links), dtype = dtype),
                               'beta': 4 * tf.ones(len(network.links), dtype = dtype)},
               trainables={'alpha': True, 'beta':True},
               capacities = [link.bpr.k for link in network.links],
               free_flow_traveltimes =[link.bpr.tf for link in network.links],
               dtype = dtype
               )


def create_tvodlulpe_model_fresno(network, dtype = _DTYPE, n_periods=1, features_Z = _FEATURES_Z, historic_g = None,
                                  performance_function = None):

    utility_parameters = UtilityParameters(features_Y=['traveltime'],
                                           features_Z= features_Z,
                                           # initial_values={
                                           #                 'traveltime': -10,
                                           #                 'tt_sd': -10, 'median_inc': 1,
                                           #                 'incidents': -1, 'bus_stops': -1, 'intersections': -1,
                                           #                 'psc_factor': 0,
                                           #                 'fixed_effect': np.zeros_like(network.links)},
                                           initial_values={
                                               'traveltime': -3.0597,
                                               'tt_sd': -3.2678, 'median_inc': 0,
                                               'incidents': -4.5368, 'bus_stops': 0, 'intersections': -3.8788,
                                               'psc_factor': 0,
                                               'fixed_effect': np.zeros_like(network.links)},

                                           signs={'traveltime': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'traveltime': True, 'tt_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           time_varying=True,
                                           dtype=dtype
                                           )

    if performance_function is None:
        # performance_function = create_bpr(network = network, dtype = dtype)
        performance_function = create_mlp(network=network, dtype=dtype)


    od_parameters = ODParameters(key='od',
                                 initial_values=network.q.flatten(),
                                 # historic_values={10: network.q.flatten()},
                                 # total_trips={0: 1e5, 1: 1e5, 2: 1e5, 9: 1e5, 10: 1e5, 11: 1e5},
                                 ods=network.ods,
                                 n_periods=n_periods,
                                 time_varying=True,
                                 trainable=True,
                                 )

    # TODO: Add option to pretrain weights
    generation_parameters = GenerationParameters(
        features_Z=['population', 'income', 'bus_stops'],
        keys=['fixed_effect_od', 'fixed_effect_origin', 'fixed_effect_destination'],
        # initial_values={'income': 1e2, 'population': 1e2, 'bus_stops': -1e2 ,
        #                 # 'fixed_effect': od_parameters.compute_generated_trips()*generation_factors
        #                 # 'fixed_effect': od_parameters.compute_generated_trips()
        #                 },
        signs={'income': '+', 'population': '+', 'bus_stops': '-'},
        trainables={'fixed_effect': True, 'income': True, 'population': True, 'bus_stops': True,
                    'fixed_effect_origin': False, 'fixed_effect_destination': False, 'fixed_effect_od': True
                    # 'fixed_effect_origin': False, 'fixed_effect_destination': True, 'fixed_effect_od': False
                    },
        # trainables={'fixed_effect': True, 'income': True, 'population': True, 'bus_stops': True},
        # trainables={'fixed_effect': False, 'income': False, 'population': False, 'bus_stops': False},
        time_varying=True,
        # historic_g = od_parameters.compute_generated_trips(),
        historic_g= historic_g,
        pretrain_generation_weights=True,
        dtype=dtype
    )

    model = NESUELOGIT(
        key='tvodlulpe',
        network=network,
        dtype=dtype,
        utility=utility_parameters,
        performance_function=performance_function,
        od=od_parameters,
        generation=generation_parameters,
        n_periods=n_periods
    )

    return model, {'utility_parameters': utility_parameters, 'generation_parameters': generation_parameters,
                   'od_parameters': od_parameters, 'performance_function': performance_function}

if run_model['equilibrium']:

    print('Equilibrium computation')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['traveltime'],
                                           features_Z= _FEATURES_Z,
                                           initial_values={'traveltime': -1, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'traveltime': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': _FIXED_EFFECT,
                                                       'traveltime': False, 'tt_sd': False, 'median_inc': False,
                                                       'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           )

    performance_function = BPR(keys=['alpha', 'beta'],
                      initial_values={'alpha': 0.15, 'beta': 4},
                      true_values={'alpha': 0.15, 'beta': 4},
                      trainables={'alpha': True, 'beta':True},
                      capacities = [link.bpr.k for link in fresno_network.links],
                      free_flow_traveltimes =[link.bpr.tf for link in fresno_network.links],
                      dtype = _DTYPE
                      )

    performance_function = MLP(n_links = len(fresno_network.links),dtype = _DTYPE)


    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * fresno_network.q.flatten(),
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    print("\nSUELOGIT equilibrium")

    suelogit = NESUELOGIT(
        key='suelogit',
        # endogenous_flows=True,
        network=fresno_network,
        dtype=_DTYPE,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        performance_function=performance_function,
        od=od_parameters
    )
    # X_train.shape
    # Y_train.shape
    train_results_dfs['suelogit'], val_results_dfs['suelogit'] = suelogit.fit(
        X_train, Y_train, X_val, Y_val,
        # generalization_error={'train': False, 'validation': True},
        optimizers=_OPTIMIZERS,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'traveltime': 0, 'flow': 0, 'eq_flow': 1},
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # print("\nTravel time based autoencoder")
    # model_0b = AETSUELOGIT(
    #     key='model_0b',
    #     # endogenous_traveltimes=True,
    #     network=fresno_network,
    #     dtype=_DTYPE,
    #     equilibrator=equilibrator,
    #     column_generator=column_generator,
    #     utility=utility_parameters,
    #     performance_function=performance_function,
    #     od=od_parameters
    # )
    #
    # train_results_dfs['model_0b'], val_results_dfs['model_0b'] = model_0b.train(
    #     X_train, Y_train, X_val, Y_val,
    #     # generalization_error={'train': False, 'validation': True},
    #     optimizers=_OPTIMIZERS,
    #     batch_size=_BATCH_SIZE,
    #     loss_weights={'od': 0, 'theta': 0, 'traveltime': 0, 'flow': 0, 'bpr': 0, 'eq_tt': 1e5},
    #     epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=val_results_dfs['suelogit'])

if run_model['odlulpe']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix (link specifics alphas and betas)')

    # _LR = 5e-1
    # _RELATIVE_GAP = 1e-5

    # performance_function = BPR(keys=['alpha', 'beta'],
    #                            initial_values={'alpha': 0.15 * tf.ones(len(fresno_network.links), dtype = _DTYPE),
    #                                            'beta': 4 * tf.ones(len(fresno_network.links), dtype = _DTYPE)},
    #                            # initial_values={'alpha': 0.15, 'beta': 4 }, # This is not numerically stable
    #                            trainables={'alpha': True, 'beta':True},
    #                            capacities = [link.bpr.k for link in fresno_network.links],
    #                            free_flow_traveltimes =[link.bpr.tf for link in fresno_network.links],
    #                            dtype = _DTYPE
    #                            )

    performance_function = MLP(n_links=len(fresno_network.links),
                               free_flow_traveltimes= [link.bpr.tf for link in fresno_network.links],
                               capacities=[link.bpr.k for link in fresno_network.links],
                               kernel_constraint = KernelConstraint(
                                   link_keys=[(link.key[0], link.key[1]) for link in fresno_network.links],
                                   capacities= [link.bpr.k for link in fresno_network.links],
                                   free_flow_traveltimes= [link.bpr.tf for link in fresno_network.links],
                                   # diagonal=True,
                                   diagonal = False,
                                   symmetric = True,
                                   adjacency_constraint= True
                               ),
                               dtype=_DTYPE)

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 ods=fresno_network.ods,
                                 trainable=True
                                 )

    utility_parameters = UtilityParameters(features_Y=['traveltime'],
                                           features_Z=_FEATURES_Z,
                                           # initial_values={
                                           #     'traveltime': -10,
                                           #     'tt_sd': -10, 'median_inc': 10,
                                           #     'incidents': -10, 'bus_stops': -10, 'intersections': -10,
                                           #     'psc_factor': 0,
                                           #     'fixed_effect': np.zeros_like(fresno_network.links)},
                                           initial_values={
                                               'traveltime': -2.8586,
                                               'tt_sd': -3.7984, 'median_inc': 0,
                                               'incidents': -0.7533, 'bus_stops': 0, 'intersections': -1.8976,
                                               'psc_factor': 0,
                                               'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'traveltime': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': _FIXED_EFFECT,
                                                       'traveltime': True, 'tt_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           dtype=_DTYPE
                                           )

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )


    generation_parameters = GenerationParameters(
        features_Z=['population','income', 'bus_stops'],
        keys=['fixed_effect_od', 'fixed_effect_origin', 'fixed_effect_destination'],
        initial_values={'income': 1e1, 'population': 1e1, 'bus_stops': -1e1 ,
                        'fixed_effect': od_parameters.compute_generated_trips()
                        # 'fixed_effect': od_parameters.compute_generated_trips()
                        },
        signs={'income': '+', 'population': '+', 'bus_stops': '-'},
        trainables={'fixed_effect': False, 'income': True, 'population': True, 'bus_stops': True,
                    'fixed_effect_origin': False, 'fixed_effect_destination': False, 'fixed_effect_od': True
                    # 'fixed_effect_origin': False, 'fixed_effect_destination': True, 'fixed_effect_od': False
                    },
        # trainables={'fixed_effect': True, 'income': True, 'population': True, 'bus_stops': True},
        # trainables={'fixed_effect': False, 'income': False, 'population': False, 'bus_stops': False},
        time_varying=True,
        historic_g=od_parameters.compute_generated_trips(),
        dtype=_DTYPE,
        pretrain_generation_weights=True,
    )

    models['odlulpe'] = NESUELOGIT(
        key='odlulpe',
        network=fresno_network,
        dtype=_DTYPE,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        performance_function=performance_function,
        od=od_parameters,
        generation=generation_parameters,
    )

    train_results_dfs['odlulpe'], val_results_dfs['odlulpe'] = models['odlulpe'].fit(
        X_train, Y_train, X_val, Y_val,
        node_data=node_data,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 1, 'theta': 0, 'traveltime': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights= _LOSS_WEIGHTS,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        alternating_optimization=_ALTERNATING_OPTIMIZATION,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        loss_metric=_LOSS_METRIC,
        epochs=_EPOCHS)
    
    plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                curves=['travel time', 'link flow'],
                                epochs_end_learning_stage = _EPOCHS['learning']
                                )

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                epochs_end_learning_stage=_EPOCHS['learning'])

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
                                yaxis_label='mape (%)')

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=val_results_dfs['odlulpe'],
                                xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                curves=['travel time', 'link flow'],
                                epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
                                yaxis_label='mape (%)')


    if models['odlulpe'].performance_function.type == 'bpr':

        plot_convergence_estimates(estimates=train_results_dfs['odlulpe'][['epoch', 'alpha', 'beta']],
                                   xticks_spacing=_XTICKS_SPACING)

        sns.displot(pd.melt(pd.DataFrame({'alpha':models['odlulpe'].performance_function.alpha, 'beta': models['odlulpe'].performance_function.beta}), var_name = 'parameters'),
                    x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)

    plot_convergence_estimates(estimates=train_results_dfs['odlulpe'].\
                           assign(rr = train_results_dfs['odlulpe']['tt_sd']/train_results_dfs['odlulpe']['traveltime'])[['epoch','rr']],
                               xticks_spacing = _XTICKS_SPACING)

    sns.displot(pd.DataFrame({'fixed_effect':np.array(models['odlulpe'].fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    plt.show()


    fig, ax = plot_convergence_estimates(
        estimates=train_results_dfs['odlulpe'].assign(
            relative_gap=np.abs(train_results_dfs['odlulpe']['relative_gap']))[['epoch', 'relative_gap']],
        xticks_spacing=_XTICKS_SPACING)

    ax.set_yscale('log')
    ax.set_ylabel('relative gap (log scale)')

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(models['odlulpe'].theta.numpy())))}")
    if models['odlulpe'].performance_function.type == 'bpr':
        print(f"alpha = {np.mean(models['odlulpe'].alpha): 0.2f}, beta  = {np.mean(models['odlulpe'].beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(models['odlulpe'].q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

    metrics_df = models['odlulpe'].compute_loss_metrics(metrics = {'mape': mape, 'mse': mse, 'r2': r2},
                                                                X = X_train, Y = Y_train).assign(dataset = 'training')
    metrics_df = pd.concat([metrics_df,
                            models['odlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                                     X=X_val, Y = Y_val).assign(dataset = 'validation')])

    with pd.option_context('display.float_format', '{:0.3g}'.format):
        print(pd.pivot(metrics_df, index = ['component', 'dataset'], columns = ['metric'])['value'])

if run_model['tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    # Total trips tvodlulpe pesuelogit:
    # Epoch 0: 6.4e+04 6.6e+04 6.3e+04 7.8e+04 7.9e+04 7.9e+04
    # Final epoch: 6.6e+04 6.6e+04 6.6e+04 6.6e+04 6.6e+04 6.6e+04
    growth_factor = 1
    # growth_factor = 7.9/6.6

    # generation_factors = growth_factor * (df.groupby(['hour'])[['counts']].mean() / df[df.hour == 16].counts.mean()). \
    #                                          reset_index().query('hour in [6,7,8, 15,16,17]')['counts'].values[:,
    #                                      np.newaxis]

    generation_factors = compute_generation_factors(period_column=XT_train[:, :, -1, None].numpy(),
                                                    flow_column=YT_train[:,:,1, None].numpy(), reference_period=10)

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generated_trips = compute_generated_trips(q = fresno_network.q.flatten()[np.newaxis,:], ods= fresno_network.ods)

    models['tvodlulpe'], _ = create_tvodlulpe_model_fresno(
        n_periods = n_periods, network = fresno_network, historic_g= generated_trips * generation_factors.values[:, np.newaxis])

    train_results_dfs['tvodlulpe'], val_results_dfs['tvodlulpe'] = models['tvodlulpe'].fit(
        XT_train, YT_train, XT_val, YT_val,
        node_data=node_data,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        # batch_size=_BATCH_SIZE,
        loss_weights= _LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        alternating_optimization=_ALTERNATING_OPTIMIZATION,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    models['tvodlulpe'].save_weights(models['tvodlulpe']._filepath_weights)

    if _PLOTS:
        # Plot heatmap with flows of top od pairs
        plot_top_od_flows_periods(models['tvodlulpe'],
                                  historic_od= fresno_network.q.flatten(),
                                  period_keys = period_keys,
                                  period_feature='hour', top_k=20)

        plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=val_results_dfs['tvodlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    curves=['travel time', 'link flow'],
                                    epochs_end_learning_stage = _EPOCHS['learning']
                                    )

        plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=val_results_dfs['tvodlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    curves=['equilibrium'],
                                    epochs_end_learning_stage=_EPOCHS['learning'])

        plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=val_results_dfs['tvodlulpe'],
                                    xticks_spacing=_XTICKS_SPACING, show_validation=True,
                                    curves=['travel time', 'link flow'],
                                    epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
                                    yaxis_label='mape (%)')

        # plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=val_results_dfs['tvodlulpe'],
        #                             xticks_spacing=_XTICKS_SPACING, show_validation=True,
        #                             epochs_end_learning_stage=_EPOCHS['learning'], prefix_metric='mape',
        #                             curves=['equilibrium'],
        #                             yaxis_label='mape (%)')

        fig, ax = plot_convergence_estimates(
            estimates=train_results_dfs['tvodlulpe'].assign(
                relative_gap=np.abs(train_results_dfs['tvodlulpe']['relative_gap']))[['epoch', 'relative_gap']],
            xticks_spacing=_XTICKS_SPACING)

        ax.set_yscale('log')
        ax.set_ylabel('relative gap (log scale)')

        plt.show()

        if models['tvodlulpe'].performance_function.type == 'bpr':

            plot_convergence_estimates(estimates=train_results_dfs['tvodlulpe'][['epoch', 'alpha', 'beta']],
                                       xticks_spacing=_XTICKS_SPACING)

            sns.displot(pd.melt(pd.DataFrame({'alpha':models['tvodlulpe'].performance_function.alpha,
                                              'beta': models['tvodlulpe'].performance_function.beta}), var_name = 'parameters'),
                        x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)

            plt.show()

        # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
        theta_df = plot_utility_parameters_periods(models['tvodlulpe'], period_keys = period_keys, period_feature='hour')

        print(theta_df)
        print(theta_df.values)

        plot_rr_by_period(models['tvodlulpe'], period_keys, period_feature='hour')

        sns.displot(pd.DataFrame({'fixed_effect': np.array(models['tvodlulpe'].fixed_effect)}),
                    x="fixed_effect", multiple="stack", kind="kde", alpha=0.8)

        plt.show()

    # models['tvodlulpe'].kappa
    print(f"theta = "
          f"{dict(zip(models['tvodlulpe'].utility.true_values.keys(), list(models['tvodlulpe'].theta.numpy())))}")
    print(f"kappa= "
          f"{dict(zip(models['tvodlulpe'].generation.features, list(np.mean(models['tvodlulpe'].kappa.numpy(), axis=0))))}")

    if models['tvodlulpe'].performance_function.type == 'bpr':
        print(f"alpha = {np.mean(models['tvodlulpe'].performance_function.alpha): 0.2f}, "
              f"beta  = {np.mean(models['tvodlulpe'].performance_function.beta): 0.2f}")

    print(f"Avg abs diff of observed and estimated OD: "
          f"{np.mean(np.abs(models['tvodlulpe'].q - fresno_network.q.flatten())): 0.2f}")

    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

    metrics_df = models['tvodlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                          X=XT_val, Y=YT_val).assign(dataset='validation')
    metrics_df = pd.concat([metrics_df,
                            models['tvodlulpe'].compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                                     X=XT_train, Y=YT_train).assign(dataset='training')])

    with pd.option_context('display.float_format', '{:0.3g}'.format):
        print(pd.pivot(metrics_df, index=['component', 'dataset'], columns=['metric'])['value'])

if run_model['tvodlulpe-kfold']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generation_factors = compute_generation_factors(period_column=XT_train[:, :, -1, None].numpy(),
                                                    flow_column=YT_train[:, :, 1, None].numpy(), reference_period=10)

    generated_trips = compute_generated_trips(q=fresno_network.q.flatten()[np.newaxis, :], ods=fresno_network.ods)

    model, _ = create_tvodlulpe_model_fresno(
        n_periods=n_periods, network=fresno_network,
        historic_g=generation_factors.values[:, np.newaxis] * generated_trips)

    metrics_kfold_df = train_kfold(
        n_splits=10,
        random_state = _SEED,
        model = model,
        X = XT_train, Y = YT_train,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        node_data = node_data,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        batch_size=1,
        epochs={'learning': 20, 'equilibrium': 10},
        # batch_size=None,
        # epochs={'learning': 3, 'equilibrium': 5}
        # epochs_print_interval= {'learning': 100, 'equilibrium': 100},
        # epochs= {'learning': 4, 'equilibrium': 5}
    )

    metrics_kfold_df.to_csv(f"./output/experiments/{datetime.now().strftime('%y%m%d%H%M%S')}_kfold_{fresno_network.key}.csv")

    # TODO: Add coefficient of variation and save experiments results, compute percentage reduction between final and initial
    with pd.option_context('display.float_format', '{:0.3g}'.format):
        print(metrics_kfold_df[metrics_kfold_df.component.isin(['flow','traveltime'])].\
              groupby(['dataset', 'component', 'metric', 'stage'])['value'].\
              aggregate(['mean', 'std']))

    plot_metrics_kfold(df = metrics_kfold_df[metrics_kfold_df.component.isin(['flow','traveltime'])])

    plt.show()

if run_model['tvodlulpe-outofsample']:

    print('\ntvodlulpe-outofsample')

    generation_factors = compute_generation_factors(period_column=XT_train[:, :, -1, None].numpy(),
                                                    flow_column=YT_train[:,:,1, None].numpy(), reference_period=10)

    n_periods = len(np.unique(XT_train[:, :, -1].numpy().flatten()))

    generated_trips = compute_generated_trips(q = fresno_network.q.flatten()[np.newaxis,:], ods= fresno_network.ods)

    reference_model, _ = create_tvodlulpe_model_fresno(
        n_periods=n_periods, network=fresno_network, historic_g=generation_factors.values[:, np.newaxis] * generated_trips)

    reference_model.build()

    # TODO: Replace with the file of a model trained on the full dataset of 2019 but with subset of links. The current
    # model comes from a kfold experiment of size 10.
    # reference_model.load_weights('output/models/230524225021_tvodlulpe_fresno.h5')

    # Model estimated with all data from 2019
    reference_model.load_weights('output/models/230525123416_tvodlulpe_fresno.h5')

    # Create model for inference
    model = create_inference_model(creation_method = create_tvodlulpe_model_fresno, reference_model=reference_model)

    model.weights

    # model.save_weights(model._filepath_weights)

    reference_model.fit(
        XT_train, YT_train,
        optimizers=_OPTIMIZERS,
        # generalization_error={'train': False, 'validation': True},
        node_data = node_data,
        # batch_size=_BATCH_SIZE,
        batch_size= None,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        equilibrium_stage=_EQUILIBRIUM_STAGE,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        # epochs=_EPOCHS
        epochs={'learning': 10, 'equilibrium':5}
    )

    # reference_model.load_node_data(node_data)
    # TODO: This should convergence immediately when the data for prediction is equal to training set
    reference_model.predict(XT_train,
                            # period_dict = reference_model.period_dict,
                            node_data=node_data,
                            loss_metric=_LOSS_METRIC,
                            optimizer=_OPTIMIZERS['equilibrium'],
                            batch_size= 1,
                            loss_weights={'eq_flow': 1},
                            threshold_relative_gap=1e-2,  # _RELATIVE_GAP,
                            epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
                            epochs=30)

    with pd.option_context('display.float_format', '{:0.3g}'.format):
        print('\n')
        training_metrics = reference_model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2},
                                                                X=XT_train, Y=YT_train)

        print(training_metrics)

    # model.summary()

    model.predict(XT_val,
                  node_data=node_data,
                  loss_metric=_LOSS_METRIC,
                  optimizer=_OPTIMIZERS['equilibrium'],
                  batch_size= 1,
                  loss_weights={'eq_flow': 1},
                  threshold_relative_gap=1e-2,  # _RELATIVE_GAP,
                  epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
                  epochs=100)

    with pd.option_context('display.float_format', '{:0.3g}'.format):
        validation_metrics = model.compute_loss_metrics(metrics = {'mape': mape, 'mse': mse, 'r2': r2}, X = XT_val, Y = YT_val)
        print(validation_metrics)

    with pd.option_context('display.float_format', '{:0.3g}'.format):
        metrics_df = pd.concat([training_metrics.assign(dataset = 'training'),
                                validation_metrics.assign(dataset = 'validation')])
        print(pd.pivot(metrics_df, index=['component', 'dataset'], columns=['metric'])['value'])

    # Bad model because it will make a prediction without training parameters
    other_model = create_tvodlulpe_model_fresno(
        historic_g=generated_trips * generation_factors.values[:, np.newaxis], network = fresno_network)[0]

    other_model.build()
    other_model.setup_period_ids(X_train=XT_val, node_data=node_data)

    # other_model.forward(X)
    with pd.option_context('display.float_format', '{:0.3g}'.format):
        print(other_model.compute_loss_metrics(metrics={'mape': mape, 'mse': mse, 'r2': r2}, X=XT_val, Y=YT_val))

    # TODO: Add benchmark model here

sys.exit()

## Write estimation results

train_results_df, val_results_df \
    = map(lambda x: pd.concat([results.assign(model = model)[['model'] + list(results.columns)]
                               for model, results in x.items()],axis = 0), [train_results_dfs, val_results_dfs])

train_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_results_{'Fresno'}.csv")
val_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_validation_results_{'Fresno'}.csv")

# ## Write predictions
#
# predictions = pd.DataFrame({'link_key': list(fresno_network.links_keys) * Y_train.shape[0],
#                             'observed_traveltime': Y_train[:, :, 0].numpy().flatten(),
#                             'observed_flow': Y_train[:, :, 1].numpy().flatten()})
#
# predictions['date'] = sorted(df[df.hour == 16].loc[df[df.hour == 16].year == 2019, 'date'])
#
# # TODO: Write predictions for TVODLULPE model
# for model in [lue,odlue,odlulpe]:
#
#     predicted_flows = model.flows()
#     predicted_traveltimes = model.traveltimes()
#
#     predictions['predicted_traveltime_' + model.key] = np.tile(predicted_traveltimes, (Y_train.shape[0], 1)).flatten()
#     predictions['predicted_flow_' + model.key] = np.tile(predicted_flows, (Y_train.shape[0], 1)).flatten()
#
# predictions.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_predictions_{'Fresno'}.csv")


## Summary of models parameters
# models = [lue,odlue,odlulpe, tvodlulpe]
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

print(results.pivot_table(index = ['parameter'], columns = 'model', values = 'values', sort=False).round(4))

# Summary of models goodness of fit

results_losses = pd.DataFrame({})
loss_columns = ['loss_flow', 'loss_tt', 'loss_eq_flow', 'loss_total']

for i, model in models.items():

    results_losses_model = model.split_results(train_results_dfs[model.key])[1].assign(model = model.key)
    results_losses_model = results_losses_model[results_losses_model.epoch == _EPOCHS['learning']].iloc[[0]]
    results_losses = pd.concat([results_losses, results_losses_model])

results_losses[loss_columns] = (results_losses[loss_columns]-1)*100

print(results_losses[['model'] + loss_columns].round(1))

## Plot of convergence toward true rr across models

# models = [lue,odlue,odlulpe, tvodlulpe]

train_estimates = {}
train_losses = {}

for i, model in models.items():
    train_estimates[model.key], train_losses[model.key] = model.split_results(results=train_results_dfs[model.key])

    train_estimates[model.key]['model'] = model.key

train_estimates_df = pd.concat(train_estimates.values())

train_estimates_df['rr'] = train_estimates_df['tt_sd']/train_estimates_df['traveltime']

estimates = train_estimates_df[['epoch','model','rr']].reset_index().drop('index',axis = 1)
estimates = estimates[estimates.epoch != 0]

fig, ax = plt.subplots(nrows=1, ncols=1)

g = sns.lineplot(data=estimates, x='epoch', hue='model', y='rr')

ax.set_ylabel('average reliability ratio')

plt.ylim(ymin=0)

plt.show()

# Plot of relibility ratio by hour for all models

reliability_ratios = plot_rr_by_period_models(models.values(), period_keys, period_feature='hour')
plt.show()

# Plot of total trips by hour for all models
plot_total_trips_models(models = models.values(), period_feature = 'hour', period_keys = period_keys)
plt.show()
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
import seaborn as sns
from datetime import datetime

# Modules from pesuelogit
from pesuelogit.visualizations import plot_heatmap_demands, plot_convergence_estimates
from pesuelogit.models import compute_rr
from pesuelogit.networks import load_k_shortest_paths, build_tntp_network
from pesuelogit.etl import get_design_tensor, add_period_id

# Internal modules
from nesuelogit.models import NESUELOGIT, ODParameters, UtilityParameters, BPR, MLP, KernelConstraint, \
    GenerationParameters, train_val_split_by_links, train_kfold, compute_generated_trips, compute_generation_factors, \
    create_inference_model, compute_benchmark_metrics
from nesuelogit.visualizations import plot_predictive_performance, plot_metrics_kfold, \
    plot_top_od_flows_periods, plot_utility_parameters_periods
from nesuelogit.metrics import mse, btcg_mse, mnrmse, mape, r2, nrmse, zscore, nmse
from nesuelogit.experiments import simulate_nesuelogit_data, simulate_features


# Seed for reproducibility
_SEED = 2023
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)


# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:',main_dir)

## Build network
network_name = 'SiouxFalls'
# network_name = 'Eastern-Massachusetts'
tntp_network = build_tntp_network(network_name=network_name)

## Read OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q=Q)

# links_df = isl.reader.read_tntp_linkdata(network_name='SiouxFalls')
# links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]
#
# # Link performance functions (assumed linear for consistency with link_cost function definion)
# tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
#                                                      'alpha': links_df.b,
#                                                      'beta': links_df.power,
#                                                      'tf': links_df.free_flow_time,
#                                                      'k': links_df.capacity
#                                                      }))

# Paths
load_k_shortest_paths(network=tntp_network, k=3, update_incidence_matrices=True)

n_links = len(tntp_network.links)
_FEATURES_Z = ['tt_sd', 's']

# n_sparse_features = 0
# features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# utility_function.add_sparse_features(Z=features_sparse)

utility_function = UtilityParameters(features_Y=['traveltime'],
                                     features_Z=_FEATURES_Z,
                                     true_values={'traveltime': -1, 'tt_sd': -1.3, 's': -3},
                                     #  true_values={'traveltime': -1, 'c': -6, 's': -3}
                                     dtype = tf.float32
                                     )

_DTYPE = tf.float32

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
                               # diagonal=True,
                               # homogenous=True
                           ),
                           depth=1,
                           dtype=dtype)

def create_bpr(network, dtype =_DTYPE):
    return BPR(keys=['alpha', 'beta'],
               initial_values={'alpha': 0.15, 'beta': 4},
               true_values={'alpha': 0.15, 'beta': 4},
               # initial_values={'alpha': 1, 'beta': 1}, # Consistent with MLP initialization
               # true_values={'alpha': 1, 'beta': 1},
               # initial_values={'alpha': 0.15 * tf.ones(len(network.links), dtype = dtype),
               #                 'beta': 4 * tf.ones(len(network.links), dtype = dtype)},
               trainables={'alpha': True, 'beta':True},
               capacities = [link.bpr.k for link in network.links],
               free_flow_traveltimes =[link.bpr.tf for link in network.links],
               dtype = dtype
               )

def create_tvodlulpe_model_siouxfalls(network, dtype=_DTYPE, n_periods=1, features_Z=_FEATURES_Z, historic_g=None,
                                      performance_function=None):
    # optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['traveltime'],
                                           features_Z=features_Z,
                                           # initial_values={'traveltime': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                           #                 'fixed_effect': np.zeros_like(network.links)},
                                           true_values={'traveltime': -1, 'tt_sd': -1.3, 's': -3},
                                           # true_values={'traveltime': [-1, -2], 'tt_sd': [-1.3, 2.6], 's': [-3, -4]},
                                           # trainables={'psc_factor': False, 'fixed_effect': False
                                           #     , 'traveltime': True, 'tt_sd': True, 's': True},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'traveltime': False, 'tt_sd': False, 's': False},
                                           time_varying=True,
                                           dtype=dtype
                                           )

    # utility_parameters.random_initializer((-1,1),['traveltime','tt_sd','s'])
    utility_parameters.random_initializer((0, 0), ['traveltime', 'tt_sd', 's'])

    if performance_function is None:
        performance_function = create_bpr(network = network, dtype = dtype)
        # performance_function = create_mlp(network = network, dtype = dtype)

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
            'fixed_effect_origin': False, 'fixed_effect_destination': False, 'fixed_effect_od': True
            # 'fixed_effect_origin': False, 'fixed_effect_destination': True, 'fixed_effect_od': False
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



# Generate synthetic node data
node_data = pd.DataFrame({'key': [node.key for node in tntp_network.nodes],
                          'income': np.random.rand(len(tntp_network.nodes)),
                          'population': np.random.rand(len(tntp_network.nodes))
                          })

# Generate data from multiple days. The value of the exogenous attributes varies between links but not between days (note: sd_x is the standard deviation relative to the true mean of traffic counts)

n_days = 300

df = simulate_features(links=tntp_network.links,
                       features_Z=_FEATURES_Z,
                       option='continuous',
                       time_variation=False,
                       range=(0, 1),
                       n_days = n_days)

df['period'] = 0
df.loc[df.timepoint >= int(2/3*n_days), 'period'] = 2

df = add_period_id(df, period_feature='period')

# period ids should 0 and 1

X = get_design_tensor(Z=df[_FEATURES_Z + ['period_id']], n_links=n_links, n_timepoints=len(df['timepoint'].unique()))

# X = exogenous_features[_FEATURES_Z + features_sparse + ['period']].values.\
#     reshape(-1,len(tntp_network.links), len(_FEATURES_Z + features_sparse )+1)

n_periods = len(np.unique(X[:, :, -1].numpy().flatten()))

generated_trips = compute_generated_trips(q=isl.networks.denseQ(Q).flatten()[np.newaxis, :],
                                          ods=tntp_network.ods)

len(tntp_network.nodes)

generation_factors = np.array([1,0.8])
# generation_factors = np.array([1,1])

model, _ = create_tvodlulpe_model_siouxfalls(network=tntp_network,
                                             n_periods=n_periods,
                                             historic_g=generation_factors[:, np.newaxis] * generated_trips)


Y = simulate_nesuelogit_data(
    model = model,
    X = X,
    batch_size=1,
    loss_metric=nmse,
    max_epochs=1000,
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-2),
    threshold_relative_gap=1e-3,
    sd_x = 0.05,
    sd_t = 0.05,
    # coverage = 0.75
)

# Check if generation factors that are computed with observed link flows match the ground truth
generation_factors = compute_generation_factors(period_column=X[:, :, -1, None].numpy(),
                                                flow_column=Y[:, :, 1, None].numpy(), reference_period=0)
print(generation_factors)

# Generate a pandas dataframe
df['traveltime'], df['counts'] = tf.unstack(Y.numpy().reshape(-1, 2),axis = 1)

df = df.drop('period_id', axis = 1)

# X = exogenous_features[_FEATURES_Z + features_sparse + ['period']].values.\
#     reshape(-1,len(tntp_network.links), len(_FEATURES_Z + features_sparse )+1)

output_file = tntp_network.key + '-link-data.csv'
output_dir = Path('input/network-data/' + tntp_network.key + '/links')
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / output_file, index=False)

# # Generate a new dataframe but asssuming a 50% coverage
#
# df_lower_coverage = simulate_suelogit_data(
#     days= list(exogenous_features.period.unique()),
#     features_data = exogenous_features,
#     equilibrator=equilibrator,
#     sd_x = 0.1,
#     sd_t = 0.1,
#     coverage = 0.5,
#     network = tntp_network)
#
# output_file = tntp_network.key + '-link-data-lower-coverage.csv'
#
# df_lower_coverage.to_csv(output_dir / output_file, index=False)
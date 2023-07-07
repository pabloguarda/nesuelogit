import isuelogit as isl
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple
from nesuelogit.models import NESUELOGIT
from nesuelogit.metrics import zscore, nmse

from isuelogit.printer import block_output

def simulate_features(n_days, time_variation = False, **kwargs) -> pd.DataFrame:
    """
    
    :param n_days: 
    :param time_variation: when True, the values of the exogenous features change between different timepoints 
    :param kwargs: 
    :return: 
    """

    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i in range(1, n_days + 1):

        if i == 1 or time_variation:
            df_day = linkdata_generator.simulate_features(**kwargs)
            df_day.insert(0, 'timepoint', i)
        else:
            df_day = df_day.assign(timepoint = i)

        df_list.append(df_day)

    df = pd.concat(df_list)

    return df


def simulate_nesuelogit_data(model: NESUELOGIT,
                             X,
                             optimizer,
                             threshold_relative_gap = 1e-5,
                             max_epochs = 100,
                             loss_metric = None,
                             batch_size = None,
                             # coverage = 1,
                             sd_x: float = 0,
                             sd_t: float = 0):

    """

    :param model:
    :param X: The last column has the period id. The remaining columns correspond to the exogenous features
    :param learning_rate:
    :param threshold_relative_gap:
    :param max_epochs:
    :param coverage:
    :param sd_x:
    :param sd_t:
    :return:
    """
    if loss_metric is None:
        loss_metric = zscore

    model.compute_equilibrium(tf.cast(X, dtype = model.dtype),
                              loss_metric= loss_metric,
                              optimizer= optimizer,
                              batch_size=batch_size,
                              loss_weights={'equilibrium': 1},
                              threshold_relative_gap= threshold_relative_gap,
                              # epochs_print_interval= _EPOCHS_PRINT_INTERVAL,
                              epochs=max_epochs)

    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))

    with block_output(show_stdout=False, show_stderr=False):
        Y_pred = model.predict(tf.cast(X, dtype = model.dtype),
                               period_dict={k: v for k, v in model.period_dict.items()},
                               loss_metric= loss_metric,
                               optimizer= optimizer,
                               batch_size=batch_size,
                               loss_weights={'equilibrium': 1},
                               threshold_relative_gap=threshold_relative_gap,  # _RELATIVE_GAP,
                               # epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
                               epochs=max_epochs)

    traveltimes, link_flows = tf.unstack(Y_pred, axis = -1)

    linkdata_generator = isl.factory.LinkDataGenerator()

    noisy_flow = linkdata_generator.add_error_counts(original_counts=link_flows.numpy(), sd_x=sd_x)
    noisy_traveltime = linkdata_generator.add_error_counts(original_counts=traveltimes.numpy(), sd_x=sd_t)

    return tf.stack([noisy_traveltime,noisy_flow], axis = 2)

# def simulate_suelogit_data(days: List,
#                            features_data: pd.DataFrame,
#                            network: TransportationNetwork,
#                            equilibrator: Equilibrator,
#                            coverage = 1,
#                            sd_x: float = 0,
#                            sd_t: float = 0,
#                            daytoday_variation = False,
#                            **kwargs):
#     linkdata_generator = isl.factory.LinkDataGenerator()
#
#     df_list = []
#
#     for i, period in enumerate(days):
#         printIterationBar(i + 1, len(days), prefix='days:', length=20)
#
#         # linkdata_generator.simulate_features(**kwargs)
#         df_period = features_data[features_data.period == period]
#
#         network.load_features_data(linkdata=df_period)
#
#         if i == 0 or daytoday_variation:
#
#             with block_output(show_stdout=False, show_stderr=False):
#                 counts, _ = linkdata_generator.simulate_counts(
#                     network=network,
#                     equilibrator=equilibrator, #{'mu_x': 0, 'sd_x': 0},
#                     coverage=1)
#
            # masked_counts, _ = linkdata_generator.mask_counts_by_coverage(
            #     original_counts=np.array(list(counts.values()))[:, np.newaxis], coverage=coverage)
#
#         counts_day_true = np.array(list(counts.values()))[:, np.newaxis]
#         counts_day_noisy = linkdata_generator.add_error_counts(
#             original_counts=masked_counts, sd_x=sd_x)
#
#         df_period['counts'] = counts_day_noisy
#         df_period['true_counts'] = counts_day_true
#
#         # Generate true travel times from true counts
#         network.load_traffic_counts(counts=dict(zip(counts.keys(),counts_day_true.flatten())))
#         df_period['true_traveltime'] =  [link.true_traveltime for link in network.links]
#
#         # Put nan in links where no traffic count data is available
#         df_period['traveltime'] = [link.true_traveltime if ~np.isnan(count) else float('nan')
#                                         for link,count in zip(network.links, masked_counts)]
#
#         df_period['traveltime'] = linkdata_generator.add_error_counts(
#             original_counts=np.array(df_period['traveltime'])[:, np.newaxis], sd_x=sd_t)
#
#         df_list.append(df_period)
#
#     df = pd.concat(df_list)
#
#     # df.groupby('link_key').agg('mean')
#
#     return df
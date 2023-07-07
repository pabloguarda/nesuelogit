import os

import matplotlib.pyplot as plt
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from  matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import collections.abc

from sklearn.linear_model import LinearRegression

plt.style.use('default')


import seaborn as sns
import scipy as sp
import pandas as pd
import numpy as np

sns.set_style('ticks')
sns.set_context('notebook')

# from isuelogit.estimation import compute_vot
# from typing import Union, Dict, List, Tuple
# from isuelogit.mytypes import Matrix
from pesuelogit.models import compute_rr
from nesuelogit.models import bpr_function, utility_parameters_periods
from nesuelogit.metrics import r2_score

import time

from typing import Union, Dict, List, Tuple
from isuelogit.mytypes import Matrix


def plot_metrics_kfold(df, metric_name: str = 'mape', sharex=True, sharey=True, **kwargs):
    # fig, axs = plt.subplots()

    sns.set_style("whitegrid")

    fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(8, 7), sharex=sharex, sharey=sharey)

    df = df.copy()
    df.loc[df.stage == 'final', 'stage'] = 'model'

    # fig, ax = plt.subplots()
    ax = axs[0, 0]
    sns.boxplot(data=df[(df.stage == 'initial') & (df.metric == metric_name)], x="component", y="value", hue='dataset',
                ax=ax, **kwargs)
    # ax.set_title(metric_name + ' before model training')
    ax.set_title('start of model training')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')

    # fig, ax = plt.subplots()
    ax = axs[0, 1]
    sns.boxplot(data=df[(df.stage == 'model') & (df.metric == metric_name)], x="component", y="value", hue='dataset',
                ax=ax, **kwargs)
    # ax.set_title(metric_name + ' after model training')
    ax.set_title('end of model training')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')

    df = df[df['stage'].isin(['benchmark', 'model'])]
    df['stage'] = pd.Categorical(df['stage'], categories=['benchmark', 'model'])

    ax = axs[1, 0]
    sns.boxplot(data=df[(df.dataset == 'training')
                        & (df.metric == metric_name) & (df.stage != 'initial')], x="component", y="value", hue='stage',
                ax=ax, palette=list(sns.color_palette("deep"))[3:5], **kwargs)
    # ax.set_title(metric_name + ' in training set')
    ax.set_title('training set')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    ax.legend(title=None)

    # fig, axs = plt.subplots()
    ax = axs[1, 1]
    sns.boxplot(data=df[(df.dataset == 'validation')
                        & (df.metric == metric_name) & (df.stage != 'initial')], x="component", y="value", hue='stage',
                ax=ax, palette=list(sns.color_palette("deep"))[3:5], **kwargs)
    # ax.axhline(df[(df.dataset == 'validation') & (df.metric == metric_name) & (df.stage == 'benchmark') ])
    # ax.set_title(metric_name + ' in validation set')
    ax.set_title('validation set')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    ax.legend(title=None)

    for ax in axs.reshape(-1):
        # ax.get_legend().remove()
        # ax.set_xlim(xmin=-100)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.get_yaxis().get_label().set_visible(True)
        ax.get_xaxis().get_label().set_visible(True)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        #ax.grid(False)

    axs[0, 1].get_yaxis().get_label().set_visible(False)
    axs[1, 1].get_yaxis().get_label().set_visible(False)

    axs[0, 0].get_xaxis().get_label().set_visible(False)
    axs[0, 1].get_xaxis().get_label().set_visible(False)

    sns.set_style("ticks")

    return fig, axs



    # axs[0, 0].yaxis.set_tick_params(which='both', labelleft=True)


def plot_parameters_kfold(df, n_cols_legend = 2, figsize = (5.5,5.5), hour_label = False, style = 'whitegrid', **kwargs):
    # bbox_to_anchor_utility = [0.27, -0.15]

    sns.set_style(style)

    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=figsize)
    bbox_to_anchor = [0.55, -0.15]

    x_label = 'period'

    if hour_label:
        x_label = 'hour'

    sns.pointplot(data=df, x=x_label, y='value', hue='parameter', ax=axs, **kwargs)

    axs.legend(loc='upper center',
                       # ncols=len(df.query("group == 'utility'").parameter.unique()),
                       ncols=n_cols_legend,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=BlendedGenericTransform(fig.transFigure, axs.transAxes))

    sns.set_style("ticks")

    return fig, axs

def plot_parameters(*args, **kwargs):
    # bbox_to_anchor_utility = [0.27, -0.15]

    return plot_parameters_kfold(style = 'ticks', *args, **kwargs)


# def plot_parameters_kfold(df, n_cols_utility_legend = 2, n_cols_generation_legend = None, figsize = (12,6),
#                           **kwargs):
#     groups = df.group.unique()
#
#     bbox_to_anchor_utility = [0.27, -0.15]
#
#     axs_generation = None
#     if len(groups) == 2:
#         fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=figsize)
#         axs_utility, axs_generation = axs
#     elif len(groups) == 1:
#         figsize = (6, 6)
#         fig, axs_utility = plt.subplots(1, 1, tight_layout=True, figsize=figsize)
#         bbox_to_anchor_utility = [0.55, -0.15]
#     else:
#         return None
#
#     if n_cols_generation_legend is None:
#         n_cols_generation_legend = len(df.query("group == 'generation'").parameter.unique())
#
#     if 'utility' in groups:
#         sns.pointplot(data=df.query("group == 'utility'"),
#                       x='period', y='value', hue='parameter', ax=axs_utility, **kwargs)
#
#         axs_utility.legend(loc='upper center',
#                            # ncols=len(df.query("group == 'utility'").parameter.unique()),
#                            ncols=n_cols_utility_legend,
#                            bbox_to_anchor=bbox_to_anchor_utility,
#                            bbox_transform=BlendedGenericTransform(fig.transFigure, axs_utility.transAxes))
#
#     if 'generation' in groups:
#         sns.pointplot(data=df.query("group == 'generation'"),
#                       x='period', y='value', hue='parameter', ax=axs_generation, **kwargs)
#
#         axs_generation.legend(loc='upper center',
#                               ncols=n_cols_generation_legend,
#                               bbox_to_anchor= [0.78, -0.15],
#                               bbox_transform=BlendedGenericTransform(fig.transFigure, axs_generation.transAxes)
#                               )
#
#     axs = [axs_utility, axs_generation]
#
#     return fig, axs




def plot_flow_interaction_matrix(flow_interaction_matrix,
                                 masking_matrix,
                                 links_ids = None, **kwargs):

    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(9, 4), sharex=True, sharey=True)
    # fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)

    if not isinstance(flow_interaction_matrix,np.ndarray):
        flow_interaction_matrix = (flow_interaction_matrix*np.eye(masking_matrix.shape[0]))

    if links_ids is None:
        links_ids = np.array(range(1, flow_interaction_matrix.shape[0]+1)).astype(str)

    cmap = sns.diverging_palette(10, 133, as_cmap=True)

    for matrix, ax in zip([masking_matrix,flow_interaction_matrix], axs):

        if kwargs.get('vmin', None) is None or kwargs.get('vmax', None) is None:
            bound = np.max(np.abs(flow_interaction_matrix))
            kwargs['vmin'], kwargs['vmax'] = -bound, bound

        matrix_df = pd.DataFrame({'link_1': pd.Series([], dtype=int),
                                  'link_2': pd.Series([], dtype=int),
                                  'weight': pd.Series([], dtype=float)})

        rows, cols = matrix.shape

        counter = 0
        for i, link_id1 in zip(range(0, rows), links_ids):
            for j, link_id2 in zip(range(0, cols), links_ids):
                matrix_df.loc[counter] = [int(link_id1), int(link_id2), matrix[(i, j)]]
                counter += 1

        matrix_df.link_1 = matrix_df.link_1.astype(int)
        matrix_df.link_2 = matrix_df.link_2.astype(int)

        matrix_pivot_df = matrix_df.pivot_table(index='link_1', columns='link_2', values='weight')

        # ax

        sns.heatmap(matrix_pivot_df, linewidth=0.5, cmap=cmap, ax = ax, **kwargs)

    for ax in axs:
        ax.set_xlabel('link')
        ax.set_ylabel('link')
        # ax.set_yticklabels(ax.get_yticklabels(), rotation = 'horizontal')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        ax.yaxis.set_tick_params(which='both', labelleft=True, rotation = 'auto')
        ax.xaxis.set_tick_params(which='both', labelbottom=True, rotation=90)


        # ax.tick_params(axis='x', rotation=90)
        #ax.tick_params(axis='y',rotation = 90)
        # ax.ticklabel_format(style = 'plain')
        # ax.locator_params(integer=True)
        # ax.get_yaxis().set_major_formatteryaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.yticks(rotation = 'horizontal')
    axs[0].set_title('Initial kernel')
    axs[1].set_title('Final kernel')

    #plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: np.math.ceil(x)))
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: np.math.ceil(x)))

    # plt.show()

    return fig, axs




def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame = None,
                                epochs_end_learning_stage: int = None,
                                xticks_spacing: int = 5,
                                show_validation=False,
                                curves=None,
                                prefix_metric='loss',
                                yaxis_label='relative mse (%)',
                                **kwargs):
    # fig, ax = plt.subplots(figsize = (5,4))

    fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)

    if curves is None:
        curves = ['travel time', 'link flow', 'equilibrium']

    if epochs_end_learning_stage is not None:
        ax.axvline(epochs_end_learning_stage, linestyle='dotted', color='black')

    patches = []

    if 'travel time' in curves:

        patches.append(mpatches.Patch(color='blue', label='link flow'))

        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_traveltime'], label="travel time", color='red',
                linestyle='-')

        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_traveltime'], label="travel time (val)", color='red',
                    linestyle='--')

    if 'link flow' in curves:

        patches.append(mpatches.Patch(color='red', label='travel time'))

        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_flow'], label="link flow", color='blue',
                linestyle='-')

        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_flow'], label="link flow (val)", color='blue',
                    linestyle='--')

    if 'equilibrium' in curves and prefix_metric + '_equilibrium' in train_losses.columns:

        patches.append(mpatches.Patch(color='gray', label='equilibrium'))

        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_equilibrium'], label="equilibrium", color='gray',
                linestyle='-')

        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_equilibrium'], label="equilibrium (val)",
                    color='gray',
                    linestyle='--')

    if prefix_metric in ['loss']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    ticks = np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + xticks_spacing, xticks_spacing)

    ax.set_xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, xticks_spacing))
    ax.set_xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max() + 2)
    ax.set_xlim(-1, None)

    # plt.ylim(ymin=0, ymax=100)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('epoch')

    # ax.set_ylabel('loss')
    ax.set_ylabel(yaxis_label)

    training_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='solid', label='training loss')
    validation_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dashed', label='validation loss')
    equilibrium_stage_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dotted',
                                        label='start of equilibrium stage')

    legend_size = 9

    if show_validation:
        legend1 = plt.legend(handles=[training_line, validation_line, equilibrium_stage_line], loc='upper center',
                             ncol=3,
                             # , prop={'size': self.fontsize}
                             bbox_to_anchor=[0.52, -0.15],
                             bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                             prop={'size': legend_size}
                             )

    else:
        legend1 = plt.legend(handles=[training_line, equilibrium_stage_line], loc='upper center',
                             ncol=2,
                             # , prop={'size': self.fontsize}
                             bbox_to_anchor=[0.52, -0.15],
                             bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                             prop={'size': legend_size})

    ax.add_artist(legend1)

    legend2 = plt.legend(handles=patches, loc='upper center', ncol=len(patches),   # , prop={'size': self.fontsize}
                         bbox_to_anchor=[0.52, -0.26],
                         bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                         prop={'size': legend_size}
                         )
    #ax.add_artist(legend2)

    # legend2 = plt.legend(handles=[train_patch, val_patch], handleheight=1e-2, loc='upper center', ncol=2#, prop={'size': self.fontsize}
    #            , bbox_to_anchor=[0.52, -0.4]
    #            , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))

    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.28)

    return fig, ax

def plot_annotate_r2(ax, x, y, intercept = False, all_metrics = False, r2 = False, rho = False):

    nas = np.logical_or(np.isnan(x.values), np.isnan(y.values))

    # slope, intercept, r, p, se = sp.stats.linregress(x = x[~nas], y = y[~nas])
    # r2 = r**2
    # # ax = plt.gca()

    # _, _, rho_val, _, _ = sp.stats.linregress(x=x[~nas], y=y[~nas])
    rho_val, p = sp.stats.pearsonr(x[~nas], y[~nas])

    reg = LinearRegression(fit_intercept=intercept).fit(X=x[~nas].values.reshape(-1,1), y=y[~nas].values.reshape(-1,1))

    r2_val = r2_score(x,y)
    intercept_value = float(reg.intercept_)
    slope = float(reg.coef_)

    if rho and not r2:
        ax.text(.05, .9, r'$\rho$={:.2f}'.format(rho_val), transform=ax.transAxes)
    if r2 and not rho:
        # ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}'.format(r2, r), transform=ax.transAxes)
        ax.text(.05, .9, r'$R^2$={:.2f}'.format(r2_val), transform=ax.transAxes)
    if r2 and rho:
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}'.format(r2_val, rho_val), transform=ax.transAxes)

    if all_metrics:
        # ax.text(.05, .8, r'$R^2$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2, slope,p),
        #         transform=ax.transAxes)
        slope, intercept, r, p, se = sp.stats.linregress(x=x[~nas], y=y[~nas])
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2_val, r, slope, p),
                transform=ax.transAxes)

    new_x = np.concatenate([np.array([0]), x])
    ax.plot(new_x, intercept_value + slope * new_x, '-', color='black')

def plot_observed_flow_vs_traveltime(model, period_col = None, observed_traveltime=None, observed_flow=None,
                            hour_label = False, all_metrics = False, **kwargs):
    hue = 'period'
    # hue = 'sign'

    if hour_label:
        hue = 'hour'

    if period_col is None:
        period_col = 1
        hue = None

    plot_data = pd.DataFrame({hue: period_col})

    plot_data[hue] = pd.Categorical(plot_data[hue],
                                    plot_data[hue].sort_values().unique())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True,
                            # sharex=True, sharey=True
                            )

    observed_flow = model.mask_observed_flow(observed_flow).numpy().flatten()
    observed_traveltime = model.mask_observed_traveltime(observed_traveltime).numpy().flatten()

    plot_data['observed_traveltime'] = observed_traveltime
    plot_data['observed_flow'] = observed_flow

    sns.scatterplot(data=plot_data, x='observed_flow', y='observed_traveltime', hue=hue, ax=ax, **kwargs)
    ax.set_xlabel('observed flow')
    ax.set_ylabel('observed travel time')


    plot_annotate_r2(ax=ax, x=plot_data['observed_flow'], y=plot_data['observed_traveltime'],
                     all_metrics=all_metrics, intercept=False, rho=True)

    ax.margins(x=0.05, y=0.05)
    ax.legend(title=hue)

    return fig, ax

def plot_flow_vs_traveltime(model, period_col = None, observed_traveltime=None, observed_flow=None,
                            hour_label = False, all_metrics = False, only_observed = False, **kwargs):

    if only_observed:
        return plot_observed_flow_vs_traveltime(model = model, period_col = period_col,
                                                observed_traveltime=observed_traveltime, observed_flow=observed_flow,
                                                hour_label = hour_label, all_metrics = all_metrics, **kwargs)

    hue = 'period'
    # hue = 'sign'

    if hour_label:
        hue ='hour'

    if period_col is None:
        period_col = 1
        hue = None

    plot_data = pd.DataFrame({'predicted_flow': model.predict_flow().numpy().flatten(),
                              'predicted_traveltime': model.predict_traveltime().numpy().flatten(),
                              hue: period_col
                              })

    plot_data[hue] = pd.Categorical(plot_data[hue],
                                           plot_data[hue].sort_values().unique())

    plot_data['sign'] = ((plot_data.predicted_flow >= 0) & (plot_data.predicted_traveltime >= 0)). \
        astype(int).map({0: 'inconsistent', 1: 'consistent'})



    if observed_flow is None and observed_traveltime is None:

        fig, axs = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
        sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue=hue, ax=axs)
        plot_annotate_r2(ax=axs, x=plot_data['predicted_flow'], y=plot_data['predicted_traveltime'],
                         all_metrics = all_metrics, intercept = False, rho = True)

    elif observed_flow is not None and observed_traveltime is not None:

        fig, axs = plt.subplots(2, 2, figsize=(9, 8), tight_layout=True,
                                # sharex=True, sharey=True
                                )

        observed_flow = model.mask_observed_flow(observed_flow).numpy().flatten()
        observed_traveltime = model.mask_observed_traveltime(observed_traveltime).numpy().flatten()

        plot_data['observed_traveltime'] = observed_traveltime
        plot_data['observed_flow'] = observed_flow
        # plot_data = plot_data.dropna()

        sns.scatterplot(data=plot_data, x='observed_flow', y='observed_traveltime', hue=hue, ax=axs[0, 0], **kwargs)
        axs[0,0].set_xlabel('observed flow')
        axs[0,0].set_ylabel('observed travel time')

        sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue=hue, ax=axs[0, 1], **kwargs)
        axs[0,1].set_xlabel('estimated flow')
        axs[0,1].set_ylabel('estimated travel time')

        sns.scatterplot(data=plot_data, x='observed_flow', y='predicted_flow', hue=hue, ax=axs[1, 0], **kwargs)
        axs[1,0].set_xlabel('observed flow')
        axs[1,0].set_ylabel('estimated flow')

        sns.scatterplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', hue=hue, ax=axs[1, 1], **kwargs)
        axs[1,1].set_xlabel('observed travel time')
        axs[1,1].set_ylabel('estimated travel time')

        # sns.regplot(data=plot_data, x='observed_flow', y='observed_traveltime', ax=axs[0, 0], **kwargs)
        # sns.regplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', ax=axs[0, 1], **kwargs)
        #
        # sns.regplot(data=plot_data, x='observed_flow', y='predicted_flow', ax=axs[1, 0], **kwargs)
        # sns.regplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', ax=axs[1, 1], **kwargs)

        plot_annotate_r2(ax = axs[0,0], x = plot_data['observed_flow'], y = plot_data['observed_traveltime'],
                         all_metrics = all_metrics, intercept = False, rho = True)
        plot_annotate_r2(ax=axs[0, 1], x=plot_data['predicted_flow'], y=plot_data['predicted_traveltime'],
                         all_metrics = all_metrics, intercept = False, rho = True)
        plot_annotate_r2(ax=axs[1, 0], x=plot_data['observed_flow'], y=plot_data['predicted_flow'],
                         all_metrics = all_metrics, intercept = False, r2 = True)
        plot_annotate_r2(ax=axs[1, 1], x=plot_data['observed_traveltime'], y=plot_data['predicted_traveltime'],
                         all_metrics = all_metrics, intercept = False, r2 = True)

        # axs[1].yaxis.set_tick_params(which='both', labelleft=True)
        # axs[1].get_yaxis().get_label().set_visible(True)
        # ax.set_ylabel('predicted_traveltime')

        # plt.ylabel('predicted_traveltime')
        # axs[1].set_yticks(axs[0].get_yticks())

        for ax in axs.reshape(-1):
            #ax.get_legend().remove()

            # ax.set_xlim(xmin=0)
            # ax.set_ylim(ymin=0)
            ax.margins(x=0.05, y=0.05)
            ax.legend(title=hue)

            # if np.sum(plot_data.sign == 'inconsistent') == 0:
            #     ax.get_legend().remove()

    return fig, axs


def plot_performance_functions(model, network, flow_range = None, marginal=False, type ='mlp',
                               selected_links=None, alpha = None, beta = None, sharey = False, **kwargs):

    if flow_range is None:
        flow_range = range(0, 20000, 100)

    traveltime_flow_df = pd.DataFrame({})
    type_pf = model.performance_function.type
    flows_shape = model.flows.numpy()[1, None].shape

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True, sharey = sharey)

    links_selected = True

    if selected_links is None:
        links_selected = False
        selected_links = range(flows_shape[1])

    capacities = np.array([link.bpr.k for link in network.links])
    free_flow_traveltimes= np.array([link.bpr.tf for link in network.links])

    if alpha is None:
        alpha = 0.15*np.ones_like(capacities)
        beta = 4 * np.ones_like(capacities)

    if marginal:

        # Link-wise marginal increase
        marginal_increase_df = pd.DataFrame({})

        for link_idx in selected_links:
            for i in flow_range:
                flows = np.zeros(flows_shape)
                flows[0, link_idx] = i

                traveltime_pf = model.traveltimes(
                    flows=np.repeat(flows, model._flows.numpy().shape[0], 0))[0, :].numpy()[link_idx]

                traveltime_bpr = bpr_function(flows=flows[0, link_idx],
                                              free_flow_traveltimes=free_flow_traveltimes[link_idx],
                                              capacities=capacities[link_idx],
                                              alpha = alpha[link_idx],
                                              beta = beta[link_idx]

                                              ).numpy().flatten()

                traveltime_bpr = model.mask_predicted_traveltimes(traveltime_bpr, k = capacities[link_idx])

                marginal_increase_df = pd.concat([marginal_increase_df,
                                                  pd.DataFrame({'flow': i,
                                                                'link': link_idx,
                                                                'traveltime_exogenous_bpr': traveltime_bpr,
                                                                'traveltime_' + str(type_pf): traveltime_pf
                                                                })])

        if links_selected:
            marginal_increase_df['link'] = pd.Categorical(marginal_increase_df['link'],
                                                          marginal_increase_df['link'].sort_values().unique())


        sns.lineplot(data=marginal_increase_df, x='flow', y='traveltime_' + str(type_pf), hue='link', ax=axs[0], **kwargs)
        sns.lineplot(data=marginal_increase_df, x='flow', y='traveltime_exogenous_bpr', hue='link', ax=axs[1], **kwargs)

        axs[0].set_title(type_pf)
        axs[1].set_title(f'exogenous bpr (alpha = {round(float(np.mean(alpha)),2)}, beta = {round(float(np.mean(beta)),2)})')

        for ax in axs:
            ax.set_ylabel('travel time')
            ax.yaxis.set_tick_params(which='both', labelleft=True, rotation='auto')

        axs[1].axes.get_yaxis().get_label().set_visible(True)

        return fig

    # Joint marginal increase
    for i in flow_range:
        flows = i * np.ones(flows_shape)

        traveltime_pf = model.traveltimes(flows=np.repeat(flows, model._flows.numpy().shape[0], 0))[0, :].numpy()

        traveltime_bpr = bpr_function(flows=flows,
                                      free_flow_traveltimes=free_flow_traveltimes,
                                      capacities=capacities,
                                      alpha=alpha,
                                      beta=beta
                                      ).numpy().flatten()

        traveltime_bpr = model.mask_predicted_traveltimes(traveltime_bpr, k = capacities)

        traveltime_flow_df = pd.concat([traveltime_flow_df,
                                        pd.DataFrame({'flow': i,
                                                      'link': np.arange(network.get_n_links()),
                                                      'traveltime_exogenous_bpr': traveltime_bpr,
                                                      'traveltime_' + str(type_pf): traveltime_pf
                                                      })])

    plot_data = traveltime_flow_df[traveltime_flow_df.link.isin(selected_links)]

    if links_selected:
        plot_data['link'] = pd.Categorical(plot_data['link'], plot_data['link'].sort_values().unique())


    sns.lineplot(data=plot_data, x='flow', y='traveltime_' + str(type_pf), hue='link', ax=axs[0], **kwargs)
    axs[0].set_title(type_pf)
    sns.lineplot(data=plot_data, x='flow', y='traveltime_exogenous_bpr', hue='link', ax=axs[1], **kwargs)
    axs[1].set_title(f'exogenous bpr (alpha = {round(float(np.mean(alpha)),2)}, beta = {round(float(np.mean(beta)),2)})')

    for ax in axs:
        ax.set_ylabel('travel time')
        ax.yaxis.set_tick_params(which='both', labelleft=True, rotation='auto')

    axs[1].axes.get_yaxis().get_label().set_visible(True)

    return fig, axs

def plot_convergence_estimates(estimates: pd.DataFrame,
                               true_values: Dict = None,
                               xticks_spacing: int = 5):
    # # Add vot
    # estimates = estimates.assign(vot=true_values.apply(compute_vot, axis=1))

    estimates = pd.melt(estimates, ['epoch'], var_name = 'parameter')

    # #Add vot
    # true_values = true_values.assign(vot=true_values.apply(compute_vot, axis=1))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(estimates['parameter'].unique())]

    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize = (5.5, 5))

    if estimates['value'].notnull().sum() > 0:
        g = sns.lineplot(data=estimates, x='epoch', hue='parameter', y='value', ax = ax)

    if true_values is not None:

        true_values = pd.Series(true_values).to_frame().T
        true_values = true_values[estimates['parameter'].unique()]

        ax.hlines(y=true_values.values,
                  xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(), colors=colors, linestyle='--')

    # ax.grid(False)

    # fig.set_size_inches(4, 3)

    plt.xticks(np.arange(estimates['epoch'].min(), estimates['epoch'].max() + 1, xticks_spacing))
    plt.xlim(xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max() + 2)

    plt.legend(prop={'size': 10})

    return fig, ax

def plot_heatmap_demands(Qs: Dict[str, Matrix],
                         subplots_dims: Tuple,
                         figsize: Tuple,
                         vmin=None,
                         vmax=None,
                         folderpath: str = None,
                         filename: str = None) -> None:
    """

    Modification of heatmap_demand function from isuelogit package

    Assume list 'Qs' has 4 elements
    """

    fig, ax = plt.subplots(*subplots_dims, figsize=figsize)

    for Q, title, axi in zip(Qs.values(), Qs.keys(), ax.flat):

        rows, cols = Q.shape

        od_df = pd.DataFrame({'origin': pd.Series([], dtype=int),
                              'destination': pd.Series([], dtype=int),
                              'trips': pd.Series([], dtype=float)})

        counter = 0
        for origin in range(0, rows):
            for destination in range(0, cols):
                # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
                od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q[(origin, destination)]]
                counter += 1

        od_df.origin = od_df.origin.astype(int)
        od_df.destination = od_df.destination.astype(int)

        od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

        # uniform_data = np.random.rand(10, 12)
        sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues", vmin=vmin, vmax=vmax, ax=axi)

        axi.set_title(title)

    plt.tight_layout()

    # plt.show()

    # fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight")

    # plt.close(fig)

    return fig, ax


def plot_top_od_flows_periods(model, period_feature, period_keys, historic_od, top_k=10):
    """
    Plot top od pairs according to the largest number of trips reported in historic OD matrix
    """

    q_df = pd.DataFrame({})

    period_dict = {v: k for k, v in model.period_dict.items()}

    for i in range(model.q.shape[0]):
        # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
        q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))

        label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
        # label_period_feature_2 = label_period_feature_1+1

        # label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
        label_period_feature = label_period_feature_1

        if label_period_feature > 12:
            label_period_feature = str(label_period_feature - 12) + 'PM'
        else:
            label_period_feature = str(label_period_feature) + 'AM'

        q_df = pd.concat([q_df, pd.DataFrame(q_dict, index=[label_period_feature])])

    q_df = q_df.transpose()

    if historic_od is not None:
        q_df.insert(loc=0, column='historic_od', value=historic_od)

    # top_q = q_df.loc[q_df.var(axis = 1).sort_values(ascending=False)[0:top_k].index].sort_index()

    top_q = q_df.loc[q_df['historic_od'].sort_values(ascending=False)[0:top_k].index]  # .sort_index()

    fig, ax = plt.subplots(1,1, figsize=(5, 4), tight_layout=True)

    sns.heatmap(top_q, linewidth=0.5, cmap="Blues", vmin=0, ax=ax)

    plt.xlabel(period_feature, fontsize=12)
    plt.ylabel('od pair', fontsize=12)

    # plt.show()

    # Plot total trips by hour
    if historic_od is not None:
        total_trips_by_hour = q_df.sum(axis=0)[1:]
    else:
        total_trips_by_hour = q_df.sum(axis=0)

    total_trips_by_hour = total_trips_by_hour.reset_index().rename(columns={'index': period_feature, 0: 'total_trips'})

    fig, ax = plt.subplots(1,1, figsize=(5, 4), tight_layout=True)

    if total_trips_by_hour.shape[0] > 1:
        g = sns.pointplot(data=total_trips_by_hour, x=period_feature, y='total_trips', ax=ax,
                          label='estimated ODs', join=False)

    else:
        g = sns.pointplot(data=total_trips_by_hour, x=period_feature, y='total_trips', ax=ax,
                          join=False)

        g.axhline(total_trips_by_hour['total_trips'].values[0], label='estimated od', linestyle='solid')

    g.axhline(q_df.sum(axis=0)[0], label='historic od', linestyle='dashed')
    plt.ylabel('total trips', fontsize=12)

    ax.legend()

    # plt.show()

    # plt.xlabel(period_feature, fontsize=12)
    # plt.ylabel('od pair', fontsize=12)

    return top_q, total_trips_by_hour


def plot_rr_by_period_models(models, period_keys, period_feature='hour', **kwargs):
    rr_by_hour_models = []
    for model_key, model in models.items():

        theta_df = pd.DataFrame({})

        period_dict = {v: k for k, v in model.period_dict.items()}

        for i in range(model.theta.shape[0]):
            theta_dict = dict(zip(model.utility.features, list(model.theta[i].numpy())))

            label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
            # label_period_feature_2 = label_period_feature_1 + 1

            # label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
            label_period_feature = label_period_feature_1

            if label_period_feature > 12:
                label_period_feature = str(label_period_feature - 12) + 'PM'
            else:
                label_period_feature = str(label_period_feature) + 'AM'

            theta_dict[period_feature + '_id'] = label_period_feature
            theta_dict[period_feature] = label_period_feature_1

            theta_df = pd.concat([theta_df, pd.DataFrame(theta_dict, index=[label_period_feature])])

        rr_df = theta_df.assign(rr=theta_df.apply(compute_rr, axis=1)).reset_index()[['rr', period_feature,
                                                                                      period_feature + '_id']]

        rr_df['model'] = model_key

        rr_by_hour_models.append(rr_df)

    rr_by_hour_models = pd.concat(rr_by_hour_models)

    rr_by_hour_models = rr_by_hour_models.sort_values([period_feature], ascending=True)

    # rr_by_hour_models['model'] = pd.Categorical(rr_by_hour_models['model'], ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout = True)

    sns.pointplot(data=rr_by_hour_models, x=period_feature + '_id', y="rr", ax=ax,
                  hue='model',
                  # markers=['o', 'v', 's', '+'],
                  # palette=["black", "black", "black", "black"],
                  **kwargs)

    plt.xlabel(period_feature)
    plt.ylabel('reliability ratio', fontsize=12)
    # ax.legend()
    # plt.legend(loc='upper left')
    # ax.get_legend().remove()

    return rr_by_hour_models


def plot_rr_by_period(model, period_keys, model_key = '', period_feature='hour'):
    plot_rr_by_period_models({model_key: model}, period_keys=period_keys, period_feature=period_feature)


def compute_total_trips_models(models, period_feature, period_keys):
    """
    Plot total trips among hours
    """

    total_trips_by_hour_models = []

    for model_key, model in models.items():

        period_dict = {v: k for k, v in model.period_dict.items()}

        q_df = pd.DataFrame({})

        for i in range(model.q.shape[0]):
            # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
            q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))

            label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
            label_period_feature_2 = label_period_feature_1 + 1

            label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
            # label_period_feature = label_period_feature_1

            q_df = pd.concat([q_df, pd.DataFrame(q_dict, index=[label_period_feature])])

        q_df = q_df.transpose()

        total_trips_by_hour = q_df.sum(axis=0).reset_index().rename(columns={'index': period_feature, 0: 'total_trips'})

        total_trips_by_hour['model'] = model_key

        total_trips_by_hour_models.append(total_trips_by_hour)

    total_trips_by_hour_models = pd.concat(total_trips_by_hour_models)

    total_trips_by_hour_models['order'] = total_trips_by_hour_models[period_feature].str.split('-').str[0].astype(int)

    total_trips_by_hour_models = total_trips_by_hour_models.sort_values('order', ascending=True)

    # total_trips_by_hour_models['model'] = pd.Categorical(total_trips_by_hour_models['model'],
    #                                                      ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])

    return total_trips_by_hour_models


def plot_total_trips_models(models, period_feature, period_keys, historic_od: np.array = None, **kwargs):

    total_trips_by_hour_models = compute_total_trips_models(models=models, period_feature=period_feature,
                                                            period_keys=period_keys)

    # Replace hours to AM/PM format

    total_trips_by_hour_models[period_feature] = total_trips_by_hour_models[period_feature].str.split('-').str[
        0].astype(str). \
        apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout = True)

    g = sns.pointplot(data= total_trips_by_hour_models, x=period_feature, y='total_trips', ax=ax,
                      hue='model',
                      # markers=['o', 'v', 's', '+'], palette=["black", "black", "black", "black"],
                      **kwargs)

    if historic_od is not None:
        g.axhline(np.sum(historic_od), linestyle='dashed', color='black', label='historic od')  #

    plt.ylabel('total trips', fontsize=12)

    ax.legend()

    # plt.legend(loc='lower left')
    plt.legend(loc='lower right', title = 'model')

    return total_trips_by_hour_models




def plot_utility_parameters_periods(model, period_keys, period_feature, include_vot=False, plot=True):

    theta_df = utility_parameters_periods(
        model, period_keys = period_keys, period_feature = period_feature, include_vot=include_vot)

    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    bound = np.nanmax(theta_df[[i for i in theta_df.columns if i != period_feature]].abs().values)

    if plot:
        fig, ax = plt.subplots(tight_layout = True)

        sns.heatmap(theta_df[[i for i in theta_df.columns if i != period_feature]].transpose(),
                    linewidth=0.5, cmap=cmap,
                    vmin=-bound, vmax=bound, ax=ax)

        plt.xlabel(period_feature, fontsize=12)
        plt.ylabel('parameter', fontsize=12)

        # plt.show()

    return theta_df

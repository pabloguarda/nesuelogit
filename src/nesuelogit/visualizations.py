import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('default')

import seaborn as sns
import scipy as sp
import pandas as pd
import numpy as np
# from isuelogit.estimation import compute_vot
# from typing import Union, Dict, List, Tuple
# from isuelogit.mytypes import Matrix
from pesuelogit.models import compute_rr
from nesuelogit.models import bpr_function
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.patches as mpatches
import time

from typing import Union, Dict, List, Tuple
from isuelogit.mytypes import Matrix


def plot_metrics_kfold(df, metric_name: str = 'mape', **kwargs):
    # fig, axs = plt.subplots()

    fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(8, 7), sharex=True, sharey=True)

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
    sns.boxplot(data=df[(df.stage == 'final') & (df.metric == metric_name)], x="component", y="value", hue='dataset',
                ax=ax, **kwargs)
    # ax.set_title(metric_name + ' after model training')
    ax.set_title('end of model training')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')

    ax = axs[1, 0]
    sns.boxplot(data=df[(df.dataset == 'training')
                        & (df.metric == metric_name)], x="component", y="value", hue='stage',
                ax=ax, **kwargs)
    # ax.set_title(metric_name + ' in training set')
    ax.set_title('training set')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')

    # fig, axs = plt.subplots()
    ax = axs[1, 1]
    sns.boxplot(data=df[(df.dataset == 'validation')
                        & (df.metric == metric_name)], x="component", y="value", hue='stage',
                ax=ax, **kwargs)
    # ax.axhline(df[(df.dataset == 'validation') & (df.metric == metric_name) & (df.stage == 'benchmark') ])
    # ax.set_title(metric_name + ' in validation set')
    ax.set_title('validation set')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')

    for ax in axs.reshape(-1):
        # ax.get_legend().remove()
        # ax.set_xlim(xmin=-100)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.get_yaxis().get_label().set_visible(True)
        ax.get_xaxis().get_label().set_visible(True)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)

    axs[0,1].get_yaxis().get_label().set_visible(False)
    axs[1,1].get_yaxis().get_label().set_visible(False)

    axs[0, 0].get_xaxis().get_label().set_visible(False)
    axs[0, 1].get_xaxis().get_label().set_visible(False)

    # axs[0, 0].yaxis.set_tick_params(which='both', labelleft=True)


def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame = None,
                                epochs_end_learning_stage: int = None,
                                xticks_spacing: int = 5,
                                show_validation=False,
                                curves=None,
                                prefix_metric='loss',
                                yaxis_label='relative mse (%)',
                                **kwargs) -> None:
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

    training_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='solid', label='training')
    validation_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dashed', label='validation')
    equilibrium_stage_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dotted',
                                        label='start of equilibrium stage')

    if show_validation:
        legend1 = plt.legend(handles=[training_line, validation_line, equilibrium_stage_line], loc='upper center',
                             ncol=3
                             # , prop={'size': self.fontsize}
                             , bbox_to_anchor=[0.52, -0.15]
                             , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))

        ax.add_artist(legend1)

    legend2 = plt.legend(handles=patches, loc='upper center', ncol=len(patches)  # , prop={'size': self.fontsize}
                         , bbox_to_anchor=[0.52, -0.26]
                         , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes)
                         )
    ax.add_artist(legend2)

    # legend2 = plt.legend(handles=[train_patch, val_patch], handleheight=1e-2, loc='upper center', ncol=2#, prop={'size': self.fontsize}
    #            , bbox_to_anchor=[0.52, -0.4]
    #            , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))

    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.28)

    # fig.show()
    # plt.show()

    # fig.savefig('output/figures/test.png')

    # ax.grid(False)

    # plt.legend(prop={'size': 8})

    return fig, ax

def plot_annotate_r2(ax, x, y, all_metrics = False):

    nas = np.logical_or(np.isnan(x.values), np.isnan(y.values))
    # r, p = sp.stats.pearsonr(x[~nas], y[~nas])
    slope, intercept, r, p, se = sp.stats.linregress(x = x[~nas], y = y[~nas])
    r2 = r**2
    # ax = plt.gca()

    if all_metrics:
        # ax.text(.05, .8, r'$R^2$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2, slope,p),
        #         transform=ax.transAxes)
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2, r, slope, p),
                transform=ax.transAxes)
    else:
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}'.format(r2, r), transform=ax.transAxes)

def plot_flow_vs_traveltime(model, observed_traveltime=None, observed_flow=None, all_metrics = False, **kwargs):
    plot_data = pd.DataFrame({'predicted_flow': model.predict_flow().numpy().flatten(),
                              'predicted_traveltime': model.predict_traveltime().numpy().flatten(),
                              })

    plot_data['sign'] = ((plot_data.predicted_flow >= 0) & (plot_data.predicted_traveltime >= 0)). \
        astype(int).map({0: 'inconsistent', 1: 'consistent'})

    if observed_flow is None and observed_traveltime is None:

        fig, axs = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
        sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue='sign', ax=axs)
        axs.get_legend().remove()

    elif observed_flow is not None and observed_traveltime is not None:

        fig, axs = plt.subplots(2, 2, figsize=(9, 8), tight_layout=True,
                                # sharex=True, sharey=True
                                )

        observed_flow = model.mask_observed_flow(observed_flow).numpy().flatten()
        observed_traveltime = model.mask_observed_traveltime(observed_traveltime).numpy().flatten()

        plot_data['observed_traveltime'] = observed_traveltime
        plot_data['observed_flow'] = observed_flow
        # plot_data = plot_data.dropna()

        # sns.scatterplot(data=plot_data, x='observed_flow', y='observed_traveltime', hue='sign', ax=axs[0, 0])
        # sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue='sign', ax=axs[0, 1])
        #
        # sns.scatterplot(data=plot_data, x='observed_flow', y='predicted_flow', hue='sign', ax=axs[1, 0])
        # sns.scatterplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', hue='sign', ax=axs[1, 1])

        sns.regplot(data=plot_data, x='observed_flow', y='observed_traveltime', ax=axs[0, 0], **kwargs)
        sns.regplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', ax=axs[0, 1], **kwargs)

        sns.regplot(data=plot_data, x='observed_flow', y='predicted_flow', ax=axs[1, 0], **kwargs)
        sns.regplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', ax=axs[1, 1], **kwargs)

        plot_annotate_r2(ax = axs[0,0], x = plot_data['observed_flow'], y = plot_data['observed_traveltime'],
                         all_metrics = all_metrics)
        plot_annotate_r2(ax=axs[0, 1], x=plot_data['predicted_flow'], y=plot_data['predicted_traveltime'],
                         all_metrics = all_metrics)
        plot_annotate_r2(ax=axs[1, 0], x=plot_data['observed_flow'], y=plot_data['predicted_flow'],
                         all_metrics = all_metrics)
        plot_annotate_r2(ax=axs[1, 1], x=plot_data['observed_traveltime'], y=plot_data['predicted_traveltime'],
                         all_metrics = all_metrics)

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

    return fig, axs


def plot_mlp_performance_functions(model, network, flow_range = None, marginal=False, selected_links=None, alpha = 0.15, beta = 4, **kwargs):

    if flow_range is None:
        flow_range = range(0, 20000, 100)

    traveltime_flow_df = pd.DataFrame({})
    type_pf = model.performance_function.type
    flows_shape = model.flows()[1, None].shape

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)

    links_selected = True

    if selected_links is None:
        links_selected = False
        selected_links = range(flows_shape[1])

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
                                              free_flow_traveltimes=[link.bpr.tf for link in network.links][link_idx],
                                              capacities=[link.bpr.k for link in network.links][
                                                  link_idx],
                                              alpha = alpha,
                                              beta = beta

                                              ).numpy().flatten()

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
        axs[1].set_title(f'exogenous bpr (alpha = {alpha}, beta = {beta})')

        return fig

    # Joint marginal increase
    for i in flow_range:
        flows = i * np.ones(flows_shape)

        traveltime_pf = model.traveltimes(flows=np.repeat(flows, model._flows.numpy().shape[0], 0))[0, :].numpy()

        traveltime_bpr = bpr_function(flows=flows,
                                      free_flow_traveltimes=[link.bpr.tf for link in network.links],
                                      capacities=[link.bpr.k for link in network.links],
                                      alpha=alpha,
                                      beta=beta
                                      ).numpy().flatten()

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
    axs[1].set_title(f'exogenous bpr (alpha = {alpha}, beta = {beta})')

    return fig, axs


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

    return fig


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

    fig, ax = plt.subplots()

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

    fig, ax = plt.subplots()

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


def plot_rr_by_period_models(models, period_keys, period_feature='hour'):
    rr_by_hour_models = []
    for model in models:

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

        rr_df['model'] = model.key

        rr_by_hour_models.append(rr_df)

    rr_by_hour_models = pd.concat(rr_by_hour_models)

    rr_by_hour_models = rr_by_hour_models.sort_values([period_feature], ascending=True)

    # rr_by_hour_models['model'] = pd.Categorical(rr_by_hour_models['model'], ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(data=rr_by_hour_models, x=period_feature + '_id', y="rr", ax=ax, join=False,
                  hue='model', markers=['o', 'v', 's', '+'], palette=["black", "black", "black", "black"])
    plt.xlabel(period_feature)
    plt.ylabel('reliability ratio', fontsize=12)
    # ax.legend()
    # plt.legend(loc='upper left')
    ax.get_legend().remove()

    return rr_by_hour_models


def plot_rr_by_period(model, period_keys, period_feature='hour'):
    plot_rr_by_period_models([model], period_keys=period_keys, period_feature=period_feature)


def plot_total_trips_models(models, period_feature, period_keys, historic_od: np.array = None):
    """
    Plot total trips among hours
    """

    total_trips_by_hour_models = []

    for model in models:

        period_dict = {v: k for k, v in model.period_dict.items()}

        q_df = pd.DataFrame({})

        for i in range(model.q.shape[0]):
            # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
            q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))

            label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature])
            label_period_feature_2 = label_period_feature_1 + 1

            label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
            # label_period_feature = label_period_feature_1

            q_df = pd.concat([q_df, pd.DataFrame(q_dict, index=[label_period_feature])])

        q_df = q_df.transpose()

        total_trips_by_hour = q_df.sum(axis=0).reset_index().rename(columns={'index': period_feature, 0: 'total_trips'})

        total_trips_by_hour['model'] = model.key

        total_trips_by_hour_models.append(total_trips_by_hour)

    total_trips_by_hour_models = pd.concat(total_trips_by_hour_models)

    total_trips_by_hour_models['order'] = total_trips_by_hour_models[period_feature].str.split('-').str[0].astype(int)

    total_trips_by_hour_models = total_trips_by_hour_models.sort_values('order', ascending=True)

    total_trips_by_hour_models['model'] = pd.Categorical(total_trips_by_hour_models['model'],
                                                         ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])

    # Replace hours to AM/PM format

    total_trips_by_hour_models[period_feature] = total_trips_by_hour_models[period_feature].str.split('-').str[
        0].astype(str). \
        apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))

    fig, ax = plt.subplots(figsize=(5, 4))

    g = sns.pointplot(data=total_trips_by_hour_models, x=period_feature, y='total_trips', ax=ax, join=False,
                      hue='model',
                      markers=['o', 'v', 's', '+'], palette=["black", "black", "black", "black"])
    if historic_od is not None:
        g.axhline(np.sum(historic_od), linestyle='dashed', color='black', label='historic od')  #

    plt.ylabel('total trips', fontsize=12)

    ax.legend()

    # plt.legend(loc='lower left')
    plt.legend(loc='upper left')

    return total_trips_by_hour_models


def plot_utility_parameters_periods(model, period_keys, period_feature, include_vot=False, plot=True):
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

    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    bound = np.nanmax(theta_df[[i for i in theta_df.columns if i != period_feature]].abs().values)

    if plot:
        fig, ax = plt.subplots()

        sns.heatmap(theta_df[[i for i in theta_df.columns if i != period_feature]].transpose(),
                    linewidth=0.5, cmap=cmap,
                    vmin=-bound, vmax=bound, ax=ax)

        plt.xlabel(period_feature, fontsize=12)
        plt.ylabel('parameter', fontsize=12)

        # plt.show()

    return theta_df

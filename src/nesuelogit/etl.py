import pandas as pd
from isuelogit.nodes import Node, NodePosition
from isuelogit.links import Link
from pesuelogit.networks import TransportationNetwork
from isuelogit.factory import NetworkGenerator
from pesuelogit.etl import get_y_tensor, get_design_tensor

from typing import Dict, List, Tuple
import tensorflow as tf

def build_network(nodes_df, links_df, crs, key=''):
    # # Read nodes data
    # nodes_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'nodes/' + 'fresno-nodes-data.csv')
    #
    # # Read link specific attributes
    # links_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'links/' 'fresno-link-specific-data.csv',
    #                        converters={"link_key": ast.literal_eval, "pems_id": ast.literal_eval})

    links_df['free_flow_speed'] = links_df['length'] / links_df['tf_inrix']

    network_generator = NetworkGenerator()

    A = network_generator.generate_adjacency_matrix(links_keys=sorted(list(links_df['link_key'].values)))

    network = TransportationNetwork(A=A, key = key)

    # Create link objects and set BPR functions and attributes values associated each link
    network.links_dict = {}
    network.nodes_dict = {}

    for index, row in links_df.iterrows():

        link_key = row['link_key']

        if 'lon' in nodes_df.columns and 'lat' in nodes_df.columns:
            # Adding gis information via nodes object store in each link
            init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
            term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

            x_cord_origin, y_cord_origin = tuple(list(init_node_row[['lon', 'lat']].values[0]))
            x_cord_term, y_cord_term = tuple(list(term_node_row[['lon', 'lat']].values[0]))

            if link_key[0] not in network.nodes_dict.keys():
                network.nodes_dict[link_key[0]] = Node(key=link_key[0],
                                                       position=NodePosition(x_cord_origin, y_cord_origin, crs=crs))

            if link_key[1] not in network.nodes_dict.keys():
                network.nodes_dict[link_key[1]] = Node(key=link_key[1],
                                                       position=NodePosition(x_cord_term, y_cord_term, crs='epsg:4326'))

        node_init = network.nodes_dict[link_key[0]]
        node_term = network.nodes_dict[link_key[1]]

        network.links_dict[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

        # Store original ids from nodes and links
        network.links_dict[link_key].init_node.id = str(init_node_row['id'].values[0])
        network.links_dict[link_key].term_node.id = str(term_node_row['id'].values[0])
        # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
        network.links_dict[link_key].id = row['id']

        network.links_dict[link_key].link_type = row['link_type']
        network.links_dict[link_key].Z_dict['link_type'] = row['link_type']

    bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                      'alpha': links_df['alpha'],
                                      'beta': links_df['beta'],
                                      'tf': links_df['tf_inrix'],
                                      # 'tf': links_df['tf'],
                                      'k': pd.to_numeric(links_df['k'], errors='coerce', downcast='float')
                                      })

    # Normalize free flow travel time between 0 and 1
    # bpr_parameters_df['tf'] = pd.DataFrame(
    #     preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf']).reshape(-1, 1)))

    network.set_bpr_functions(bprdata=bpr_parameters_df)

    # To correct problem with assignment of performance

    return network

def get_tensors_by_year(df: pd.DataFrame, features_Z: List[str], links_keys: List[str]) \
        -> Tuple[Dict[str, tf.Tensor],Dict[str, tf.Tensor]]:

    n_links = len(df.link_key.unique())
    X,Y = {}, {}

    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]

        # n_dates, n_hours = len(df_year.date.unique()), len(df_year.hour.unique())
        #
        # n_timepoints = n_dates * n_hours
        n_timepoints = len(df_year[['date', 'hour']].drop_duplicates())

        # Order of data should comply with the internal order of links in the network given by links_dict
        df_year['link_key'] = pd.Categorical(df_year['link_key'], links_keys)

        df_year = df_year.sort_values(['period','link_key'])

        traveltime_data = get_y_tensor(y=df_year[['tt_avg']], n_links=n_links, n_timepoints=n_timepoints)
        flow_data = get_y_tensor(y=df_year[['counts']], n_links=n_links, n_timepoints=n_timepoints)

        Y[year] = tf.concat([traveltime_data, flow_data], axis=2)

        X[year] = get_design_tensor(Z=df_year[features_Z + ['period_id']], n_links=n_links, n_timepoints=n_timepoints)

    return X, Y
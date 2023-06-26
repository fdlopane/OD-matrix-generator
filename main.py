'''
This module generates an OD cost matrix (in minutes) among origins and destinations provided as a point shp.
The example considers Oxford MSOA centroids as origins and destination.
To run the code for a different city/region, change the input shp file and modify the "Case_Study_Area" variable.

Author:
Dr Fulvio D. Lopane
The Bartlett Centre for Advanced Spatial Analysis
University College London
'''

import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np

# Inputs:
origins = "./input-data/Oxford-MSOA-centroids.shp"

# Outputs:
OD_cost_matrix = "./outputs/cost_matrix.csv"
output_folder_path = "./outputs/Graph_shp" # to export the network shapefile

# Variables:
Case_Study_Area = ['Oxford'] # Road network

def flows_map_creation(OD, area): # Using OSM

    PoI = nx.read_shp(OD) # Shp must be in epsg:4326 (WGS84)

    X = ox.graph_from_place(area, network_type='drive')
    crs = X.graph["crs"]
    print('Graph CRS: ', crs)
    print()
    ox.plot_graph(X) # Plots the road network graph

    X = X.to_undirected()

    # Add edge speeds (km per hour) to graph as new speed_kph edge attributes:
    X = ox.speed.add_edge_speeds(X, fallback=50)

    # Add edge travel time (seconds) to graph as new travel_time edge attributes:
    X = ox.speed.add_edge_travel_times(X)

    # Calculate the origins and destinations for the shortest paths algorithms to be run on OSM graph
    # OD_list = list of network nodes that are the closest to the PoI
    # Nodes_map = dictionary: {closest network node ID: (PoI coordinates)}
    OD_list, Nodes_map = calc_shortest_paths_ODs_osm(PoI, X)
    TOT_count = len(OD_list)

    Dist_dict = {} # Dictionary containing shortest paths: {origin: {destination: travel_time}}

    for n, i in enumerate(OD_list):
        print("Shortest paths - iteration ", n + 1, " of ", TOT_count)
        Dist_dict[i] = {}
        for j in OD_list:
            sp = nx.shortest_path_length(X, source=i, target=j, weight='travel_time', method='dijkstra')
            Dist_dict[i][j] = round(sp/60.0, 2) # convert travel time from seconds to minutes

    # Convert the 2-level dictionary into a Pandas DataFrame:
    # Look here for info: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    Dist_df = pd.DataFrame.from_records(
        [
            (level1, level2, leaf)
            for level1, level2_dict in Dist_dict.items()
            for level2, leaf in level2_dict.items()
        ],
        columns=['origin', 'destination', 'travel_time']
    )

    Dist_df["origin_coord"] = ""
    Dist_df["destination_coord"] = ""

    # Add origins and destinations coordinates columns
    for i in Nodes_map.keys(): # loop through keys
        Dist_df.loc[Dist_df['origin'] == i, 'origin_coord'] = str(Nodes_map[i])
        Dist_df.loc[Dist_df['destination'] == i, 'destination_coord'] = str(Nodes_map[i])

    print(Dist_df)

    # Convert the DataFrame into a matrix:
    OD_matrix = pd.pivot_table(Dist_df, values="travel_time", index="origin_coord", columns="destination_coord", aggfunc=np.sum, margins=True)

    # Export to csv:
    # OD_matrix.to_csv(OD_cost_matrix, header=False, index=False)
    OD_matrix.to_csv(OD_cost_matrix)

    # save network graph to shapefile
    ox.save_graph_shapefile(X, filepath=output_folder_path)

def calc_shortest_paths_ODs_osm(zones_centroids, network):
    # For each zone centroid, this function calculates the closest node in the OSM graph.
    # These nodes will be used as origins and destinations in the shortest paths calculations.
    Nodes_map = {}
    list_of_ODs = []
    for c in zones_centroids:
        graph_clostest_node = ox.nearest_nodes(network, c[0], c[1], return_dist=False)
        list_of_ODs.append(graph_clostest_node)
        Nodes_map[graph_clostest_node] = c
    return list_of_ODs, Nodes_map

# Run the function:
flows_map_creation(origins, Case_Study_Area)
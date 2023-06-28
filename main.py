'''
COST MATRIX GENERATOR
This module generates a OD cost matrices (in minutes) for private (cars) and public transport.
Origins and destinations need to be provided as a point shp.
The example considers Oxford MSOA centroids as origins and destinations.
To run the code for a different city/region, change the input files (zones, centroids and GTFS)
and modify the "Case_Study_Area" variable.

Author:
Dr Fulvio D. Lopane
The Bartlett Centre for Advanced Spatial Analysis
University College London

The public transport matrix generation is heavily based on Richard Milton's methodology also available here:
https://github.com/maptube/Harmony
'''
print()
print("COST MATRIX GENERATOR")
print()

# Import modules
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import sys
import os
from os import path
import geopandas as gpd
from QUANT.PublicTransportNetwork import PublicTransportNetwork

# Inputs:
origins = "./input-data/Oxford_MSOA_centroids_WGS84.shp"
zonesDataFilename = './input-data/zones_data.csv' # NOTE: zonei must start at 1 and go in ascending order (this will be the index of rows and columns of Cij)
zoneBoundariesShapefilenameWGS84 = './input-data/Oxford_MSOA_boundaries_WGS84.shp' # Zones boundaries shapefile
GTFSDir = './input-data/GTFS_data' # GTFS data directory

# Outputs:
outputsDir = 'outputs' # outputs directory
OD_cost_matrix_private = "./outputs/Cij_private.csv" # Private transport travel cost matrix
output_folder_path = "./outputs/Graph_roads_shp" # To export the driving network shapefile
graphML_public_Filename = 'outputs/graph_public_transport.graphml' # Network graph built from GTFS files
graphVertexPositionsFilename = 'outputs/vertices_public_transport.csv' # Lat/lon of nodes in graphml file above
zoneCentroidLookupFilename = 'outputs/zone_centroid_lookup.csv' # Closest graph vertex to zone centroids
cijPublicTransportFilename = 'outputs/Cij_public.csv' # Public transport travel cost matrix

# Variables:
Case_Study_Area = ['Oxford'] # Road network
# Walking node fixup distance is used to connect network nodes that are walkable e.g. where there are two bus routes
# but none of the stops are shared, then this puts a walking link between network segments to make them connected
# - lower is better as it tends to add in lots of additional network links
walkFixupDistMetres = 500

# Create output directory if it doesn't already exist:
if not path.exists(outputsDir):
    os.mkdir(outputsDir)

################################################################################
# Driving cost matrix creation
################################################################################
def flows_map_creation(OD, area): # Using OSM
    PoI = nx.read_shp(OD) # Shp must be in epsg:4326 (WGS84)

    X = ox.graph_from_place(area, network_type='drive')
    crs = X.graph["crs"]
    print('Graph CRS: ', crs)
    print()
    # ox.plot_graph(X) # Plots the road network graph

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
    # OD_matrix.to_csv(OD_cost_matrix_private, header=False, index=False)
    OD_matrix.to_csv(OD_cost_matrix_private)

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
print("PRIVATE TRANSPORT COST MATRIX")
print()
if not path.exists(OD_cost_matrix_private):
    print("File %s does not exist, creating new" % OD_cost_matrix_private)
    print()
    flows_map_creation(origins, Case_Study_Area)
else:
    print("File %s exists, skipping creation" % OD_cost_matrix_private)
    print()

################################################################################
# Public transport cost matrix creation
################################################################################
print("PUBLIC TRANSPORT COST MATRIX")
print()
if path.exists(cijPublicTransportFilename):
    print("File %s exists, skipping creation" % cijPublicTransportFilename)
    print()
else:
    print("File %s does not exist, creating new" % cijPublicTransportFilename)
    print()

    # Load the zones_data.csv - contains mapping between zonei number and zone code string
    dfZonesData = pd.read_csv(zonesDataFilename, dtype={'zone':str})
    # NOTE: use the zone field as the area key

    print("Building Public Transport Network (PTN) from GTFS files")
    print()
    print("Initialise PTN")
    print()

    ptn = PublicTransportNetwork()

    # Make GraphML network file from GTFS data
    if not path.exists(graphML_public_Filename):
        print('File %s does not exist, so creating new' % graphML_public_Filename)
        print()
        # Tram = 0, Subway = 1, Rail = 2, Bus = 3, Ferry = 4, CableCar = 5, Gondola = 6, Funicular = 7
        ptn.initialiseGTFSNetworkGraph([GTFSDir], {'*'}) # {1,3,800,900}?
        count = ptn.FixupWalkingNodes(walkFixupDistMetres)

        print('After fixup walking nodes: ', str(ptn.graph.number_of_nodes())," vertices and ", str(ptn.graph.number_of_edges()), " edges in network.")
        print()
        ptn.writeGraphML(graphML_public_Filename)
        ptn.writeVertexPositions(graphVertexPositionsFilename) # note: you also need the vertex position map, otherwise you don't know where the graphml vertices are
    else:
        # Load exsting graph for speed
        print('File %s exists, skipping creation' % graphML_public_Filename)
        print()
        ptn.readGraphML(graphML_public_Filename)

    # Centroid lookup
    print('Loading zone codes from %s', origins)
    print()

    # ZoneLookup is the mapping between the zone code numbers and the MSOA code and lat/lon zonecentroids
    # zones_data.csv: zone = E02005946
    # shapefile: msoa11cd = E02005946
    ZoneLookup = gpd.read_file(origins, dtype={'msoa11cd': str})
    ZoneLookup['msoa11cd'] = ZoneLookup['msoa11cd'].astype(str)

    if not path.exists(zoneCentroidLookupFilename):
        print('File %s does not exist, creating new' % zoneCentroidLookupFilename)
        print()
        #use reproject into wgs84 as that's what gtfs is in
        CentroidLookup = ptn.FindCentroids(ZoneLookup, zoneBoundariesShapefilenameWGS84, 'msoa11cd')
        # Save it:
        ptn.saveCentroids(CentroidLookup, zoneCentroidLookupFilename)
    else:
        print('File %s exists, skipping creation' % zoneCentroidLookupFilename)
        print()
        # Load centroids for speed:
        CentroidLookup = ptn.loadCentroids(zoneCentroidLookupFilename)

    # Shortest paths calculations
    # Make a matrix for the results
    N = len(dfZonesData) # Number of zones
    Cij = np.zeros(N*N).reshape(N, N) # i,j are the object id
    Cij.fill(-1) # Use -1 as no data value

    # Make a lookup here of zonei->zonecode and zonei->vertexi
    # dfZonesData contains:
    # |zonei| zone     | msoa11nm             | Easting         | Northing
    # |0    | E02005976| South Oxfordshire 019| 474910.866999999| 179996.014

    zonei_to_zonecode = {} # Mapping between zone i number and string zone code
    zonei_to_vertexi = {} # Mapping between zone i number and closest vertex on network to centroid
    vertex_to_zonei = {} # Mapping between graph vertex and closest zone i zone (DESIGNATED CENTROID ONLY)
    for index, row in dfZonesData.iterrows():
        zonei = row['zonei']
        zonecode = row['zone']
        if zonecode in CentroidLookup: # Otherwise, there is no centroid and no possibility of a shortrst path cost
            gvertex = CentroidLookup[zonecode]
            zonei_to_zonecode[str(zonei)] = zonecode
            zonei_to_vertexi[str(zonei)] = gvertex
            vertex_to_zonei[gvertex] = str(zonei)

    # Shortest paths calculation:
    print("Running shortest paths")
    print()

    # result = nx.all_pairs_dijkstra_path_length(graph,weight='weight') #58s
    result = nx.all_pairs_shortest_path_length(ptn.graph) #57.5s

    # NOTE: result: key is the origin vertex, data contains a map of destination vertex with time
    # nxgraph is returning a generator, which means we have to cycle through the data in the order
    # they specify: i is origin and j is destination
    for keyi, data in result:
        # key is EVERY vertex in the data and we only want selected vertices
        if keyi in vertex_to_zonei:
            # key is a vertex designated the closest to a centroid, so fill in this zone's data
            zonei = vertex_to_zonei[keyi]
            count = 0
            print('SSSP: Origin = ', zonei, ', Destination = ', keyi, end='')
            for keyj in data:
                if keyj in vertex_to_zonei:
                    # We've got a designated centroid vertex for the destination
                    zonej = vertex_to_zonei[keyj]
                    Cij[int(zonei)-1, int(zonej)-1] = data[keyj] # cost NOTE: this puts the element (1,1) in the position (0,0)
                    count += 1
            print(' count=', count) # This is how many destinations are matched for each origin

    # Save cij (cost matrix)
    with open(cijPublicTransportFilename,'w') as f:
        for i in range(0,N):
            for j in range(0,N):
                f.write(str(Cij[i,j]))
                if j!=N-1:
                    f.write(', ')
            f.write('\n')

    # Analyse the data
    sum = 0
    min = sys.float_info.max
    max = 0
    missingcount = 0
    datacount = 0
    for i in range(0, N):
        for j in range(0, N):
            value = Cij[i, j]
            if value == -1:
                missingcount += 1
            else:
                datacount += 1
                sum += value
                if value >= max:
                    max = value
                if value <= min:
                    min = value

    print('Cij stats: mean = ', round(sum/datacount, 2),
          ', missingcount = ', missingcount,
          ', datacount = ', datacount,
          ', max = ', max,
          ', min = ', min)

print('Program finished.')
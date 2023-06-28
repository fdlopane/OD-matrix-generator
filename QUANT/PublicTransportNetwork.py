"""
PublicTransportNetwork.py
Copy of https://github.com/maptube/UMaaS

Wrapper for a network built from GTFS data
Copy of 
"""

import sys
import networkx as nx
from scipy.spatial import KDTree, cKDTree
#from sklearn.neighbors import KDTree
#from sklearn.neighbors import KNeighborsClassifier
#geopandas?
import geopandas as gpd
from shapely.geometry import Point, Polygon
#from rtree import index
import math
import time
import timeit
import ctypes as c

from QUANT.GTFSUtils import GTFSUtils
#import graphserver.nvGraph as nvGraph


class PublicTransportNetwork:

################################################################################
# gtfs properties
    #public AdjacencyGraph<string, WeightedEdge<string>> graph; //using QuickGraph structures
    #public Dictionary<Edge<string>, double> costs; //this is the distance weight list for edges

    #public KdTree<string> kdtree; //this is for finding the closest graph vertex to each msoa centroid

    
    # Return the number of vertices in the graph.
    #public int VertexCount {
    #    get { return graph.VertexCount;  }
    #}

    
    # Return the number of edges in the graph.
    #public int EdgeCount
    #{
    #    get { return graph.EdgeCount; }
    #}

    def __init__(self):
        self.graph = nx.Graph()
        self.vertices = {}
        self.vidx = {} #map between vertex number (to match kdtree) and vertex string (to match graph)
        #self.costs
        self.kdtree = None

################################################################################


    """
    Build the graph of the bus network which we can then use to run queries on.
    @param name="GTFSDirectories" (string [])
    @param name="AllowedRouteTypes" (boolean []) Bitset of RouteTypes which will be included in the following order: Tram=0, Subway=1, Rail=2, Bus=3, Ferry=4, CableCar=5, Gondola=6, Funicular=7</param>
    """
    def initialiseGTFSNetworkGraph(self, GTFSDirectories, AllowedRouteTypes):
        #graph = new AdjacencyGraph<string, WeightedEdge<string>>(); //using QuickGraph structures
        self.graph = nx.MultiDiGraph() #we need directed and parallel edges
        #costs = new Dictionary<Edge<string>, double>(); //this is the distance weight list for edges

        #kdtree = new KdTree<string>(); //this is for finding the closest graph vertex to each msoa centroid
        kdtreedata = [] #because we have to make the kdtree index in the constructor, we need to generate a list of [node_id,lat,lon] for every vertex
        #self.kdtree = index.Index() #rtree index

        self.vertices = {}
        for dir in GTFSDirectories:
            print("scanning dir "+dir)
            #Dictionary<string, List<GTFS_ODLink>> links = new Dictionary<string, List<GTFS_ODLink>>();
            #Dictionary<string, GTFSStop> vertices = new Dictionary<string, GTFSStop>();
            links, localvertices = GTFSUtils.ExtractRunlinks(dir, AllowedRouteTypes) #NOTE: make sure the Allowed route types is set correctly i.e. (rail,ferry) or (bus,ferry)
            #now have to update self.vertices with (stop code, GTFS node data) so that we have the lat/lon to make the spatial index from - AND need to merge this between GTFS dirs in top loop
            #self.vertices.update(localvertices) #update() doesn't work - why??? #map of stop code to GTFSFile.GTFSStop object, giving the lat lon
            for code, value in localvertices.items(): #do it the hard way instead - map of stop code to GTFSFile.GTFSStop object, giving the lat lon
                self.vertices[code]=value

            #now build the graph
            #NOTE: each links.value is a list of the same link, just for different times - we need to average them
            for link in links.values():
                try:
                    Stop_O = localvertices[link[0].O]
                    Stop_D = localvertices[link[0].D]
                    total_secs = 0
                    count = 0
                    MinSeconds = sys.float_info.max
                    MaxSeconds = 0
                    for timedlink in link:
                        if (timedlink.Seconds < 0):
                            print('PublicTransportNetwork::InitialiseGTFSNetworkGraph: Error, negative runlink time (made positive): ' + timedlink.O + ' ' + timedlink.D + ' ' + timedlink.Seconds)
                            #continue;
                            #you could potentially do abs(timedlink.Seconds) as it looks like they've just been put in the wrong way around?
                            timedlink.Seconds = abs(timedlink.Seconds) #make it positive and continue - needed to do this for gbrail, must have a consistent error which causes an island in the data?
                        #end if
                        total_secs += timedlink.Seconds
                        if (timedlink.Seconds < MinSeconds):
                            MinSeconds = timedlink.Seconds
                        if (timedlink.Seconds > MaxSeconds):
                            MaxSeconds = timedlink.Seconds
                        count+=1
                    #end for timed_link in link
                    AverageSecs = total_secs / count
                    if (AverageSecs < 0):
                        print('Error: negative weight: ' + AverageSecs + ' ' + link[0].O + ' ' + link[0].D)
                    #endif
#set hard minimum limit for runlink to 30 seconds
#                        if ((count==0)||(AverageSecs<30)) AverageSecs = 30.0f;

                    #add an edge (unidirectional as it's an adjacency graph)
                    #TODO: nxgraph code to go here
                    #print('Add edge ',Stop_O.Code,Stop_D.Code)
                    self.graph.add_edge(Stop_O.Code,Stop_D.Code,weight=AverageSecs)
                    #e = WeightedEdge<string>(Stop_O.Code, Stop_D.Code, AverageSecs)
                    #graph.AddVerticesAndEdge(e)
                    #costs.Add(e, AverageSecs)
                    #and add a spatial index entry for finding closest msoa centroids
                    #self.kdtree.Insert(Coordinate(Stop_O.Lon, Stop_O.Lat), Stop_O.Code)
                    #kdtreedata.append([Stop_O.Lon, Stop_O.Lat])
                    #self.kdtree.insert(Stop_O.Code, (Stop_O.Lon, Stop_O.Lat))
                except Exception as e:
                    print('Error: (missing vertex?) ' + link[0].O + ' ' + link[0].D + ' ' + str(e))
                #end try except
            #end for link in links
        #end for dir

        #build kd tree spatial index in a one shot operation using the data in self.vertices from the GTFS
        #As the kd tree can only return nodes as index positions in this original data update, what I do is
        #to create a self.vidx map (vertex index) between the id number as used in the kd tree index and the
        #vertex code as used in the gtfs. This means that when you do a kd tree search and it returns ids of
        #0, 54 and 99, then vidx[0], vidx[54] and vidx[99] will give you the gtfs vertex codes for the stops.
        #NOTE: self.vertices still contains the data indexed by gtfs code e.g. vertices["6290WB05"] gets you
        #the gtfs data object containing the lat lon.
        kdtreedata = []
        self.vidx = {}
        for idx, v in enumerate(self.vertices):
            kdtreedata.append([self.vertices[v].Lon,self.vertices[v].Lat])
            self.vidx[idx]=self.vertices[v].Code
        #end for
        #now make the kdtree index in a one shot operation
        print("kdtree length = ",len(kdtreedata))
        self.kdtree = cKDTree(kdtreedata) #NOTE: this is assuming that the node index numbers in networkx graph are the same order as the nodes were created

        print('PublicTransportNetwork::InitialiseGTFSNetworkGraph:: ', self.graph.number_of_nodes(), ' vertices and ', self.graph.number_of_edges(), ' edges in network')

################################################################################

    # <summary>
    # With a bus graph, you might get two bus nodes that are walkable, but aren't linked up because they don't share nodes. This procedure uses the spatial index to find nodes that are close and
    # adds a walking link between then based on a fixed walking speed.
    # PRE: must have called InitialiseGTFSNetworkGraph first.
    # Setting MaxWalkDistance to 2000M results in an additional 3 million edges and that's just for London.
    # 500M seems a better option.
    # TODO: need a way to avoid adding a route between two nodes where there already is a bus link i.e. need to just add walking links between different bus routes, not along bus routes. The route info isn't in here though.
    # TODO: a second alternative might be to route all bus nodes within walking distance to the MSOA centroid which you would have to add as a new node. You still need a stop within 500M, but you could make the radius bigger. No mulitple bus routes though.
    # </summary>
    # <param name="MaxWalkDistance">Maximum walking distance to join up nearby vertices in METRES</param>
    #returns number of edges added
    def FixupWalkingNodes(self,MaxWalkDistance):
        WalkSpeed = 1.39 #This is ms-1, but it's 5kmh or 3.1mph according to Google
        #const float MaxWalkDistance = 2000; //in metres - THIS WAS 500m for buses, changed for rail
        a = 6378137.0 #WGS84 semi-major
        circumference = 2 * math.pi * a
        box = MaxWalkDistance / circumference * 360.0 #search box width based on fraction of earth radius - OK, it's an approximation, so add a bit for safety
        box = box * 1.2

        print('Fixup walking nodes MaxWalkDistance=',MaxWalkDistance,' box=', box)

        # count = 0
        # env = (-180,180,-90,90) #there must be a better way of querying everything?
        # for idx in self.kdtree.query(env):
        #     lat = G.nodes[idx]
        #     lon = G.nodes[idx]
        #     float lat = (float)node.Y, lon = (float)node.X;
        #     foreach (KdNode<string> dnode in kdtree.Query(new Envelope(lon-box,lon+box,lat-box,lat+box))) #the box just needs to cover the MaxWalkDistance, it doesn't matter if it's too big
        #     {
        #         if (node.Data != dnode.Data) #check it's not the same node
        #         {
        #             dist = GTFSUtils.GreatCircleDistance(lat, lon, (float)dnode.Y, (float)dnode.X);
        #             #dist in metres, so set 500M radius around node
        #             if (dist < MaxWalkDistance)
        #             {
        #                 WeightedEdge<string> e = new WeightedEdge<string>(node.Data, dnode.Data, dist/WalkSpeed); #connect two existing nodes with a directed edge - note that we don't add any new vertices, just an edge
        #                 graph.AddEdge(e);
        #                 costs.Add(e, dist / WalkSpeed);
        #                 count+=1
        #             }
        #         }
        #     }
        # }

        #OK, let's do this differently - query all pairs of coordinates that are within the set distance, then go through each and do a better distance test before connecting them up
        #NOTE: I don't think this is anything like as efficient as the original code
        count = 0
        pairs = self.kdtree.query_pairs(box)
        for pair in pairs:
            #print(pair)
            #print(self.kdtree.data[pair[0]],self.kdtree.data[pair[1]])
            #print(self.vidx[pair[0]],self.vidx[pair[1]])
            #print(list(self.graph.nodes)[0])
            #n1 = self.kdtree.data[pair[0]] #NOTE these are (lon,lat)
            #n2 = self.kdtree.data[pair[1]]
            node1 = self.vertices[self.vidx[pair[0]]] #this is a GTFSFile.GTFSStop - NOTE: lat/lon should match with n1,n2
            node2 = self.vertices[self.vidx[pair[1]]]
            #print('n1 n2 n1id n2id',n1,n2,n1id.Code,n1id.Lon,n1id.Lat,n2id.Code,n2id.Lon,n2id.Lat)
            #NOTE: kdtree is made from self.vertices, which came from the gtfs stops. This can have nodes that aren't part of
            #the graph as they haven't been referenced in the gtfs stop_times file which contains the runlinks. The short
            #answer is that we need to check that the node ids are actually in the graph and ignore them if not.
            if node1.Code in self.graph.nodes and node2.Code in self.graph.nodes:
                if pair[0]==pair[1]:
                    print("Error: nodes are the same")
                dist = GTFSUtils.GreatCircleDistance(node1.Lat,node1.Lon,node2.Lat,node2.Lon) #and these are lat,lon
                if (dist<MaxWalkDistance):
                    links = nx.neighbors(self.graph,node1.Code)
                    if not node2 in links: #test whether this node is already connected to the other one
                        self.graph.add_edge(node1.Code, node2.Code, weight = dist/WalkSpeed)
                        count+=1
                        #print("FixupWalkingNodes added: ", node1.Code, node2.Code, dist, dist/WalkSpeed)
            #endif


        print('Fixup walking nodes finished. Added ', count, ' new edges.')
            
        return count
    
################################################################################

    # /// <summary>
    # /// As it doesn't work if you try and pick the nearest bus vertex to an msoa centroid, this function finds all the bus vertex points in the msoa and picks a middle one.
    # /// PRE: MUST have called InitialiseGTFSNetworkGraph first to set up the GTFS graph as it uses the spatial index of points (kdtree).
    # /// RWM: NOTE: generalised this from MSOA only to work with any shapefile. This required added a field name for the area code.
    # /// RWM2: NOTE: also changed the code so that it skips and warns of an area in the shapefile not in the zonecodes file - this is for the Manchester and London subsets
    # /// RMW3: also changed it to work with a geopandas dataframe and removed zonei, working instead
    # /// with the area code string all through. ZoneCodesDF must link to the boundary shapefile
    # /// using the AreaKeyField.
    # /// </summary>
    # /// <param name="ZoneCodesDF">ZoneCodes dataframe: areakey, point geom</param>
    # /// <param name="ZoneAreaKeyField"></param>
    # /// <param name="ShapefileFilename"></param>
    # /// <returns>A lookup between the MSOA (or other shapefile geometry e.g. LSOA) areakey and the unqiue string identifier of its calculated centroid bus stop</returns>
    def FindCentroids(self, ZoneCodesDF, ShapefileFilename, ShapefileAreaKeyField):
        #print(ZoneCodesDF.head())
        #print(ZoneCodesDF[ShapefileAreaKeyField])

        Result = {}

        shapefile = gpd.read_file(ShapefileFilename,dtype={ShapefileAreaKeyField:str}) #dtype doesn't work - see below
        shapefile[ShapefileAreaKeyField] = shapefile[ShapefileAreaKeyField].astype(str) #force areakey to string
        #print(shapefile.head()+"\n")
        for idx, f in shapefile.iterrows():
            #print(f)
            #for this feature, get the centroid, which we use as the origin
            areakey = f[ShapefileAreaKeyField] #was "MSOA11CD" for MSOA
            # print(f['geometry'].centroid)
            #areaname = f["MSOA11NM"] #not needed

            if areakey not in ZoneCodesDF[ShapefileAreaKeyField].values:
                print("WARN: area " + areakey + " in shapefile, but not in the zonecodes table - skipped. This will occur when processing subsets of areas from a main shapefile containing every possible area.")
                continue
            
            #int(dfPrimaryPopulation.loc[dfPrimaryPopulation['msoaiz'] == msoa_iz,'zonei'])
            Rowi = ZoneCodesDF.loc[ZoneCodesDF[ShapefileAreaKeyField]==areakey]
            # print(Rowi)
            # print(Rowi["geometry"].values[0])
            #Zonei = Rowi["zonei"]
            #CentroidLat = Rowi["lat"]
            #CentroidLon = Rowi["lon"]
            point = Rowi["geometry"].values[0]
            CentroidLat = point.y
            CentroidLon = point.x
            # print("CentroidLat=",CentroidLat)
            # print("CentroidLon=",CentroidLon)
            env = f['geometry'].envelope
            #this whole section is designed to get the centroid point and max radius distance of envelope corners so that we can do the kd tree ball query below
            #the original code did this a lot more easily with a proper kd tree implementation and an envelope query
            cx = env.centroid.x
            cy = env.centroid.y
            dist2=0
            for point in env.exterior.coords:
                # print("point=",point)
                dx = (point[0]-cx)
                dy = (point[1]-cy)
                d = dx*dx+dy*dy
                if d>dist2:
                    dist2 = d
            dist=math.sqrt(dist2)
            # print("cx=",cx,"cy=",cy,"r=",dist)

        #     List<KdNode<string>> nodes = (List<KdNode<string>>)kdtree.Query(f.Geometry.EnvelopeInternal);
            nodes = self.kdtree.query_ball_point([cx,cy],r=dist)
            #print(nodes)
            Lat = 0
            Lon = 0
            count = 0
            MaxOutDegree = 0
        #     KdNode<string> MaxOutDegreeNode = null;
            for node in nodes: #note that node is a kd tree index into the points originally used to create the indes
                graphnode = self.vertices[self.vidx[node]] #use lookup between kd tree and graph vertices to get the graph node with the actual geographic points
                #print("graphnode=",graphnode,graphnode.Code)
                #graph node is a gtfs stop point
                P = Point(graphnode.Lon,graphnode.Lat)
                if P.within(f['geometry']):
                    Lat += graphnode.Lat
                    Lon += graphnode.Lon
                    count+=1
                    #and look at the number of out edges for this node which is within the MSOA in order to find the maximum, which might be a better metric...
                    #O = self.graph.OutDegree(node.Data)
                    if (graphnode.Code in self.graph): #why on earth would a node be missing from the graph structure?
                        O = len(self.graph[graphnode.Code]) #TODO: check - is this right?????
                        if O > MaxOutDegree:
                            MaxOutDegree = O
                            MaxOutDegreeNode = graphnode
                        #end if
                    #else:
                    #    print("Graph node ",graphnode.Code," no edges data - skipped")
                    #endif
                #end if
            #end for node in nodes
            if count > 0:
                Lat /= count
                Lon /= count
                
                minDist2 = sys.float_info.max
                MinNode = None
                for node in nodes:
                    graphnode = self.vertices[self.vidx[node]]
                    dx = graphnode.Lon - Lon
                    dy = graphnode.Lat - Lat
                    dist2 = dx * dx + dy * dy
                    if (dist2 < minDist2):
                        minDist2 = dist2
                        MinNode = graphnode
                    #end if 
                #end for node in nodes

                print("PublicTransportNetwork::FindCentroids: "+areakey + ","
                    + str(CentroidLat) + "," + str(CentroidLon) + ","
                    + str(Lat) + "," + str(Lon) + ","
                    + str(MinNode.Lat) + "," + str(MinNode.Lon) + ","
                    + MinNode.Code +","+str(MaxOutDegree)+","+MaxOutDegreeNode.Code+","+str(count)
                )
                #we have options here - either return the closest node to the centroid of the stops within the MSOA (MinNode.data)
                #OR return the node within the MSOA with the most out edges (MaxOutDegreeNode)
                #Result.Add(areakey, MinNode.Data); //closest to centroid
                Result[areakey] = MaxOutDegreeNode.Code #max out edges
            else:
                #print("PublicTransportNetwork::FindCentroids: "+ areakey +", Error"); // + " " + areaname)
                print("PublicTransportNetwork::FindCentroids: " + areakey + ","
                    + str(CentroidLat) + "," + str(CentroidLon) + ","
                    + str(Lat) + "," + str(Lon) + ","
                    + "0" + "," + "0" + "," + "0" + "," + "0" + "," + "0" + "," + str(count)
                )
            #end if count>0
        
        #end for f in shapefile
        
        return Result

################################################################################

    """
    <summary>
    Test of an existing QuickGraph network being converted to the CUDA nvGraph format and Single Source Shortest Path run.
    PRE: needs and existing QuickGraph structure, so this.graph must be populated.
    </summary>
    <param name="ZoneCodes"></param>
    <param name="CentroidLookup"></param>
    <param name="OriginAreaKeys">NOT USED! List containing the origin of where to start from to get to all of the ZoneCodes in the list (arriving at their CentroidLookup nodes) e.g. "E02000001" is City of London</param>
    <returns>Dictionary<string,float> of MSOA area destination and time to reach from the origin in seconds.</returns>
    """
    def TestCUDASSSP(self, ZoneCodes, CentroidLookup): #RWM removed OriginAreaKeys
        #NOTE:
        #for APSP, set vertex_numsets=7201
        #then sssp_1 becomes an array of 7201 x sssp_1
        #vertex_dim needs to be an array into each sssp_1 block of data (and vertex_dimT)
        #NOTE: is vertex_dim ever actually used????
        #then you run nvGraph.nvgraphSssp(handle, graph, 0,  source_vert_h, 0) with the final zero as 0..7201, which puts results into each index block of the sssp_1 result
        Results = {} #new Dictionary<string, float>();
        N = len(ZoneCodes) #assuming 0..len(ZoneCodes)-1 zones
        #matrix = np.array([np.zeros(N),np.zeros(N)])

        n = self.graph.number_of_nodes()
        nnz = self.graph.number_of_edges()
        vertex_numsets = 1
        edge_numsets = 1
        #dimension arrays for the data
        weights = [0.0] * nnz
        destination_offsets = [0] * (n+1)
        source_indices = [0] * nnz
        #fill the arrays from the NetworkX structure
        #first, make a lookup between vertex name (string) and vertex number (int) for the nvGraph arrays. You could potentially do this in one step, but with more lookups.
        VIndex=0
        VNumLookup = {} #<string,int>
        for VName in self.graph.nodes:
            VNumLookup[VName] = VIndex
            VIndex+=1
        #endfor
        #OK, now have a name to index lookup, so build the data. Unfortunately, it's all backwards and QuickGraph doesn't have a graph.InEdges() method similar to graph.OutEdges().
        #Make a lookup of vertex in edges by walking the entire edge list.
        InEdges = [-1] * n
        for i in range(0,n): InEdges[i] = []
        for e in self.graph.edges:
            Dest = VNumLookup[e[1]] #edges are a list of (source,target) tuples [0]=source, [1]=target
            InEdges[Dest].append(e)
        #endfor
        #now we've got a list of destination nodes and the edges leading to them, we can build the nvGraph data
        count = 0
        for d in range(0,n):
            destination_offsets[d] = count
            for e in InEdges[d]:
                s = VNumLookup[e[0]] #if it doesn't find the name then it crashes - something's gone badly wrong
                #TODO: here, the weights need changing and check the indices
                source_indices[count] = s
                #NOTE: although the edge is a triple (source,target,weight?), the third value is always zero. You have to do it as in the line below:
                w=self.graph.edges[e]['weight']
                weights[count] = w
                count+=1
            #endfor
        #endfor
        destination_offsets[n] = count #there's an extra final n+1 vertex that you need to set
            
        #now build a list of MSOA code to vertex number which we want to route to and from (centroid)
        CUDAZoneVertexLookup = {} #new Dictionary<string, int>();
        for k, rowi in ZoneCodes.items():
            AreaKeyi = rowi["areakey"] #OK, it's also 'k' anyway
            if AreaKeyi in CentroidLookup:
                Vertexi = CentroidLookup[AreaKeyi] #this is the nearest node to the msoa centroid
                Zonei = rowi["zonei"]
                CUDAVertex = VNumLookup[Vertexi]
                CUDAZoneVertexLookup[AreaKeyi] = CUDAVertex
                #System.Diagnostics.Debug.WriteLine(AreaKeyi + "," + Zonei + "," + CUDAVertex);
            #endif
        #endfor

        #that's the data prepared, now do the real work

        #Init arrays and other data - we're only every going to be using one vertex num set and one edge num set, so let's try and make it look elegant
        print("Running SSSP: n=" + str(n) + ", nnz=" + str(nnz))
        sssp_1 = [0.0] * n * 2
        sssp_1_seq = c.c_float * len(sssp_1)
        sssp_1_h = sssp_1_seq(*sssp_1)

        vertex_dimT = [nvGraph.cudaDataType.CUDA_R_32F.value, nvGraph.cudaDataType.CUDA_R_32F.value]
        vertex_dimT_seq = c.c_int * len(vertex_dimT)
        vertex_dimT_h = vertex_dimT_seq(*vertex_dimT)
        edge_dimT = [nvGraph.cudaDataType.CUDA_R_32F.value]
        edge_dimT_seq = c.c_int * len(edge_dimT)
        edge_dimT_h = edge_dimT_seq(*edge_dimT)
        
        weights_seq = c.c_float * len(weights)
        weights_h = weights_seq(*weights)
        destination_offsets_seq = c.c_int*len(destination_offsets)
        destination_offsets_h = destination_offsets_seq(*destination_offsets)
        source_indices_seq = c.c_int*len(source_indices)
        source_indices_h = source_indices_seq(*source_indices)


        handle = nvGraph.nvgraphHandle_t()
        handle_p = c.pointer(handle)
        handle_p.contents = handle
        graph = nvGraph.nvgraphDescr_t()
        graph_p = c.pointer(graph)
        graph_p.contents = graph
            
        nvGraph.check_status("nvgraphCreate",nvGraph.nvgraphCreate(handle_p)) #now we create the graph with a graph pointer handle for the return
        nvGraph.check_status("nvgraphCreateGraphDescr",nvGraph.nvgraphCreateGraphDescr(handle, graph_p)) #and then do the same with a graph descriptor handle
        
        CSC_input = nvGraph.nvgraphCSCTopology32I_st()
        CSC_input.nvertices = n
        CSC_input.nedges = nnz
        CSC_input.destination_offsets = destination_offsets_h
        CSC_input.source_indices = source_indices_h

        # Set graph connectivity and properties (tranfers)
        nvGraph.check_status("nvgraphSetGraphStructure",nvGraph.nvgraphSetGraphStructure(handle, graph, c.pointer(CSC_input), nvGraph.nvgraphTopologyType.CSC_32.value))
        nvGraph.check_status("nvgraphAllocateVertexData",nvGraph.nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT_h))
        nvGraph.check_status("nvgraphAllocateEdgeData",nvGraph.nvgraphAllocateEdgeData(handle, graph, edge_numsets, edge_dimT_h))
        nvGraph.check_status("nvgraphSetEdgeData",nvGraph.nvgraphSetEdgeData(handle, graph, weights_h, 0, nvGraph.nvgraphTopologyType.CSC_32.value))
        # Solve
        start = time.process_time()
        start2 = time.clock()
        validcount=0
        for areakey in ZoneCodes:
            if areakey in CUDAZoneVertexLookup:
                validcount+=1
                source_vert = c.c_int(CUDAZoneVertexLookup[areakey]) #rwm was OriginAreaKeys[i]
                source_vert_h = c.pointer(source_vert)
                nvGraph.check_status("nvgraphSssp",nvGraph.nvgraphSssp(handle, graph, 0,  source_vert_h, 0))
                # Get and print result
                nvGraph.check_status("nvgraphGetVertexData",nvGraph.nvgraphGetVertexData(handle, graph, sssp_1_h, 0, nvGraph.nvgraphTopologyType.CSC_32.value))
                #todo: here you need to get the data out of the sssp_1_h data and write it back to the matrix i.e. a subset of the origin to all other nodes
            #endif
        #endif
            
        secs = time.process_time()-start #actually, both of these are in seconds!
        secs2 = time.clock() - start2
        print("SSSP Elapsed seconds = " + str(secs)+" "+str(secs2),"validcount=",validcount)
        for DestAreaKey,d in CUDAZoneVertexLookup.items():
            #DestAreaKey = vertex code and d= vertex number in CUDA structure
            #print(source_vert + " -> " + DestAreaKey + " " + str(sssp_1[d]))
            Results[DestAreaKey] = sssp_1_h[d]
        #endfor
        #print(list(sssp_1_h))
        

        #Clean up - all variables are python managed
        nvGraph.check_status("nvgraphDestroyGraphDescr",nvGraph.nvgraphDestroyGraphDescr(handle, graph))
        nvGraph.check_status("nvgraphDestroy",nvGraph.nvgraphDestroy(handle))

        return Results

################################################################################

    def writeGraphML(self,filename):
        #nx.write_graphml_lxml(self.graph, filename) # this one fails with NoneType does not contain Element
        nx.write_graphml_xml(self.graph, filename) # this one uses a different xml writer and works!

    #TODO: does this work? Doesn't read centroids or kdtree
    def readGraphML(self,filename):
        self.graph = nx.read_graphml(filename)

################################################################################

    def writeVertexPositions(self,filename):
        #this writes out the lat/lon position of each graph node
        with open(filename,"w") as f:
            f.write("vertex_name,lat,lon\n")
            for code, value in self.vertices.items():
                f.write(code+","+str(value.Lat)+","+str(value.Lon)+"\n")
            #endfor
        #endwidth

################################################################################

    def saveCentroids(self,centroids,filename):
        with open(filename,"w") as f:
            f.write("areakey,vertex\n")
            for ak, code in centroids.items():
                f.write(ak+","+code+"\n")
            #endfor
        #endwith

################################################################################

    def loadCentroids(self,filename):
        #todo: this is a conversion from the c#, maybe convert to dataframes?
        centroids = {}
        with open(filename,"r") as f:
            f.readline() #skip the header line
            lines = f.readlines()
            for line in lines:
                fields = line.split(',')
                if (len(fields)==2):
                    centroids[fields[0]]=fields[1].strip() #you get a trailing \n otherwise
                #endif
            #endfor
        #endwith
        return centroids

################################################################################

"""
GTFSUtils.py
Copy of https://github.com/maptube/UMaaS
"""

import sys
import glob
import zipfile
import math

from QUANT.GTFSFile import *

"""
Class representing an origin destination link between two points on a GTFS network. The time in seconds between the O and D is used for the runlink time.
"""
class GTFS_ODLink:
    
    def __init__(self,O,D,Seconds):
        self.O = O
        self.D = D
        self.Seconds = Seconds

################################################################################

"""
GTFS utilities
"""
class GTFSUtils:
    """
    Splits a string containing comma separated data and returns the elements as an array of strings.
    Quotes can be used around items containing a comma.
    These quotes are removed from the string array that is returned.
    
    <param name="line">The CSV line string to be split</param>
    <returns>Array of strings containing the comma separated elements in the CSV file line</returns>
    """
    @staticmethod
    def ParseCSVLine(line):
        Items = []
        Current = ""
        Quote = False
        for ch in line:
            if ch==',':
                    if not Quote:
                        Items.append(Current.strip())
                        Current = ""
                    else:
                        Current += "," #comma inside a quote
                    break
            elif ch=='"':
                Quote = not Quote
                break
            else:
                Current += ch
                break
            #end if ch==','
        #end for ch in line
        Items.append(Current.strip()) #add trailing item - even if last char was a comma, we still want a null on the end

        return Items

################################################################################


    """
    Take a directory containing GTFS files and write out a file containing the lat lons of the unique stop point locations.
    TODO: you could do this from the GTFSFile class now.
    
    @param name="InDirectory" string
    @param name="OutFilename" string
    """
    @staticmethod
    def ExtractStopPoints(InDirectory, OutFilename):
        StopPoints = [] # Dictionary<string,float[]>()

        files = glob.glob(InDirectory+'*.zip')
        for Filename in files:
            print('Processing: ' + Filename)
            with zipfile.ZipFile(Filename,'r') as zip:
                #stops.txt
                #stop_id,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url
                #9400ZZLUHAW1,"Wealdstone, Harrow & Wealdstone Station",,51.5923316813,-0.3354040827,,
                #9400ZZLUKEN2,"Kenton (London), Kenton",,51.5822158642,-0.3171684662,,
                with zip.open('stops.txt') as stops:
                    Line = stops.readLine() #skip header
                    while not stops.eof():
                        Line = stops.readLine()
                        Fields = self.ParseCSVLine(Line)
                        Name = Fields[0]
                        try:
                            Lat = float(Fields[3])
                            Lon = float(Fields[4])
                            if not StopPoints.ContainsKey(Name):
                                StopPoints.Add(Name, [ Lat, Lon ])
                            #end if
                        except:
                            print('Error: ' + Filename + ', ' + Line)
                        #end try except
                    #end while not stops.eof
                #end with zip.open
            #end with ZipFile
        #end for Filename in files

        #now write out the file
        with open(OutFilename,'w') as writer:
            writer.writeLine('stop_id,stop_lat,stop_lon')
            for KVP in StopPoints:
                writer.writeLine(KVP.Key + ',' + KVP.Value[0] + ',' + KVP.Value[1])
            #end for
        #end with open

        print('Processed ' + len(StopPoints) + ' stop points from ' + InDirectory)

################################################################################

    # <summary>
    # Returns metres.
    # </summary>
    # <param name="lat1"></param>
    # <param name="lon1"></param>
    # <param name="lat2"></param>
    # <param name="lon2"></param>
    # <returns></returns>
    @staticmethod
    def GreatCircleDistance(lat1, lon1, lat2, lon2):
        #taken from here: http://www.movable-type.co.uk/scripts/latlong.html
        #double R = 6371e3; // metres
        R = 6378137.0 #metres
        Phi1 = lat1 * math.pi / 180.0
        Phi2 = lat2 * math.pi / 180.0
        DeltaPhi = (lat2-lat1) * math.pi / 180.0
        DeltaLambda = (lon2-lon1) * math.pi / 180.0
            
        a = math.sin(DeltaPhi/2) * math.sin(DeltaPhi/2.0) + math.cos(Phi1) * math.cos(Phi2) * math.sin(DeltaLambda/2.0) * math.sin(DeltaLambda/2.0)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
        d = R * c
        return d

################################################################################


    # <summary>
    # Overload of ExtractRunLinks to enable the links and vertices to be extracted directly for building a graph straight off the gtfs files.
    # NOTE: this only extracts bus and ferry runlinks by design.
    # TODO: make the route type configurable so you could do tubes as a separate case?
    # NOTE: vertices are built from the stops file, so will contain vertices that aren't in the stop_times file
    # </summary>
    # <param name="InDirectory"></param>
    # <param name="AllowedRouteTypes">NO! Changed to a map of values. Bitset of allowed route types in following order [ Tram=0, Subway=1, Rail=2, Bus=3, Ferry=4, CableCar=5, Gondola=6, Funicular=7 ]</param>
    # <param name="links"></param> returns
    # <param name="vertices"></param> returns
    @staticmethod
    def ExtractRunlinks(InDirectory, AllowedRouteTypes):
        #key is "origin|destination", value is a list of all the OD links for this segment so we can compute an average time and frequency of service
        links = {} #new Dictionary<string, List<GTFS_ODLink>>();
        vertices = {} #new Dictionary<string, GTFSStop>();

        files = glob.glob(InDirectory+'/*.zip')

        for Filename in files:
            print('Processing: ' + Filename)
            gtfs = GTFSFile(Filename)
            #The stop times are keyed on the tripid, so each gtfs.StopTimes dictionary value contains a list GTFSStopTimes, which is a list of stops and times for a single TripId
            for stops in gtfs.StopTimes.values():
                #route type detection code - BUS and FERRY ONLY
                #find the trip id for this stop time, which gives the the route id, which give me the route and the route type - horribly inefficient - should walk the routes and trips
                TripId = stops[0].trip_id
                RouteId = gtfs.Trips[TripId].route_id
                Route = gtfs.Routes[RouteId]
                #if ((Route.route_type != GTFSRouteType.Bus) && (Route.route_type != GTFSRouteType.Ferry)) continue;
                if '*' not in AllowedRouteTypes: # a * in the route types means all allowed
                    if not Route.route_type in AllowedRouteTypes: #otherwise, check this type is on the allowed list
                        #check that this route type is on the list - if not then exit
                        #print('Skip route type=', Route.route_type)
                        continue
                #end if
                #end of route type detection code

                #starting a new timed service, which is a list of stops
                LastStop = None
                for Stop in stops:
                    #build OD links from this i.e. last stop to this stop
                    if LastStop != None:
                        #and OD link has an O and D key, plus a runtime
                        #How do you cope with different arrival and departure times? Always take the departure?
                        runlink = Stop.departure_time - LastStop.departure_time
                        OD = GTFS_ODLink(LastStop.stop_id, Stop.stop_id, runlink.seconds)
                        key = LastStop.stop_id + '|' + Stop.stop_id
                        if not key in links:
                            links[key] = []
                        links[key].append(OD)
                    #end if
                    LastStop = Stop
                #end for stop in stops
            #end for stops in gtfs.stoptimes.values
            #and concatenate the stops data from this file into a master list for later
            for stop in gtfs.Stops.values():
                if not stop.Code in vertices:
                    vertices[stop.Code] = stop #of course, this doesn't handle stops with the same code and different data...
            #end for
        #end for filename in files
        print('Extracted ', len(links), ' distinct OD segments.')
        return links, vertices

################################################################################


    # <summary>
    # Passed in a directory containing gtfs zip files, process out runlinks for origin destination pairs and write them out.
    # Use the trips.txt file to get the journey paths and the stop_times.txt file to get the runlinks. Stop locations in stops.txt file.
    # TODO: you could stick the hour of day into the OD link to see how it varied over the course of the day.
    # </summary>
    # <param name="InDirectory"></param>
    # <param name="OutFilename"></param>
#     def ExtractRunlinks(self, InDirectory, OutFilename, AllowedRouteTypes):
#         links = {}
#         vertices = {}
#         GTFSUtils.ExtractRunlinks(InDirectory, AllowedRouteTypes, links, vertices)

#         #now you have to write the file out
#         with open(OutFilename, 'w') as writer:
#             writer.writeLine('code_o,lat_o,lon_o,code_d,lat_d,lon_d,average_secs,min_seconds,max_seconds,count,distance_metres,average_speed_ms-1')
#             #NOTE: each links.value is a list of the same link, just for different times - we need to average them
#             for link in links.Values:
#                 try:
#                     Stop_O = vertices[link[0].O]
#                     Stop_D = vertices[link[0].D]
#                     total_secs = 0
#                     count = 0
#                     MinSeconds = sys.float_info.max
#                     MaxSeconds = 0
#                     for timedlink in link:
#                         total_secs += timedlink.Seconds
#                         if (timedlink.Seconds < MinSeconds):
#                             MinSeconds = timedlink.Seconds
#                         if (timedlink.Seconds > MaxSeconds):
#                             MaxSeconds = timedlink.Seconds
#                         count+=1
#                     #end for timedlink
#                     AverageSecs = total_secs / count
# #Set hard limit of 30 seconds for minimum link time
# #                        if ((count == 0) || (AverageSecs < 30)) AverageSecs = 30;
#                     DistMetres = self.GreatCircleDistance(Stop_O.Lat, Stop_O.Lon, Stop_D.Lat, Stop_D.Lon)
#                     #some of the runlinks are zero seconds, so need to be careful about average speed
#                     AverageSpeedMS = -1
#                     if (AverageSecs>0):
#                         AverageSpeedMS = DistMetres / AverageSecs
#                     writer.WriteLine(Stop_O.Code + ',' + Stop_O.Lat + ',' + Stop_O.Lon + ',' + Stop_D.Code + ',' + Stop_D.Lat + ',' + Stop_D.Lon
#                         + ',' + AverageSecs + ',' + MinSeconds + ',' + MaxSeconds + ',' + count + ',' + DistMetres + ',' + AverageSpeedMS)
#                 except Exception as e:
#                     print('Error: (missing vertex?) ' + link[0].O + ' ' + link[0].D + ' '+e)
#                 #end try except
#             #end for link in links
#         #end with open


################################################################################


    # <summary>
    # Convert the data from ExtractRunLinks into a shapefile containing lines linking the origin and destination nodes.
    # </summary>
    # <param name="InDirectory">List of CSV files to process</param>
    # <param name="OutFilename"></param>
    # public static void ConvertODRunlinksToShapefile(string [] InCSVFiles, string OutFilename)
    # {
    #     GeometryFactory gf = new GeometryFactory();
    #     List<Feature> fc = new List<Feature>();
    #     foreach (string file in InCSVFiles)
    #     {
    #         using (TextReader reader = File.OpenText(file))
    #         {
    #             string Line = reader.ReadLine(); //skip header line
    #             while ((Line = reader.ReadLine()) != null)
    #             {
    #                 #code_o,lat_o,lon_o,code_d,lat_d,lon_d,average_secs,min_seconds,max_seconds,count,distance_metres,average_speed_ms-1
    #                 #9400ZZLUEAC1,51.49582,-0.1009803,9400ZZLULBN1,51.49852,-0.1111693,120,120,120,1882,767.7943,6.398285
    #                 string[] Fields = Line.Split(new char[] { ',' });
    #                 string code_o = Fields[0];
    #                 float lat_o = Convert.ToSingle(Fields[1]);
    #                 float lon_o = Convert.ToSingle(Fields[2]);
    #                 string code_d = Fields[3];
    #                 float lat_d = Convert.ToSingle(Fields[4]);
    #                 float lon_d = Convert.ToSingle(Fields[5]);
    #                 float AverageSecs = Convert.ToSingle(Fields[6]);
    #                 float MinSeconds = Convert.ToSingle(Fields[7]);
    #                 float MaxSeconds = Convert.ToSingle(Fields[8]);
    #                 float Count = Convert.ToSingle(Fields[9]);
    #                 float DistanceMetres = Convert.ToSingle(Fields[10]);
    #                 float AverageSpeedMS = Convert.ToSingle(Fields[11]);
    #                 Feature f = new Feature();
    #                 Coordinate OriginPoint = new Coordinate(lon_o,lat_o);
    #                 Coordinate DestPoint = new Coordinate(lon_d,lat_d);
    #                 f.Geometry = gf.CreateLineString(new Coordinate[] { OriginPoint, DestPoint });
    #                 f.Attributes = new AttributesTable();
    #                 f.Attributes.AddAttribute("origin", code_o);
    #                 f.Attributes.AddAttribute("dest", code_d);
    #                 f.Attributes.AddAttribute("AverageSecs", AverageSecs);
    #                 f.Attributes.AddAttribute("MinSecs", MinSeconds);
    #                 f.Attributes.AddAttribute("MaxSecs", MaxSeconds);
    #                 f.Attributes.AddAttribute("Count", Count);
    #                 f.Attributes.AddAttribute("DistM", DistanceMetres);
    #                 f.Attributes.AddAttribute("AvSpeedMS", AverageSpeedMS);
    #                 fc.Add(f);
    #             }
    #         }
    #     }
    #     ShapeUtils.WriteShapefile(OutFilename, fc);
    # }


################################################################################

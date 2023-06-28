"""
GTFSFile.py
Copy of https://github.com/maptube/UMaaS
"""
from enum import Enum
from datetime import datetime, timedelta
import zipfile
import pandas as pd

################################################################################

class GTFSRouteType(Enum):
        Tram=0
        Subway=1
        Rail=2
        Bus=3
        Ferry=4
        CableCar=5
        Gondola=6
        Funicular=7

################################################################################


# <summary>
# From routes.txt
# </summary>
class GTFSRoute:
    def __init__(self, RouteId, AgencyId, RouteShortName, RouteLongName, RouteDesc, RouteType):
        self.route_id = RouteId
        self.agency_id = AgencyId
        self.route_short_name = RouteShortName
        self.route_long_name = RouteLongName
        self.route_desc = RouteDesc
        self.route_type = RouteType

################################################################################


# <summary>
# From stops.txt
# </summary>
class GTFSStop:
    def __init__(self, Code, Name, Lat, Lon):
        self.Code = Code
        self.Name = Name
        self.Lat = Lat
        self.Lon = Lon

################################################################################


# <summary>
# From trips.txt
# </summary>
class GTFSTrip:
    def __init__(self, route_id, service_id, trip_id, trip_headsign):
        self.route_id = route_id
        self.service_id = service_id
        self.trip_id = trip_id
        self.trip_headsign = trip_headsign

################################################################################


# <summary>
# From stop_times.txt
# </summary>
class GTFSStopTime:
    def __init__(self, trip_id, arrival_time, departure_time, stop_id, stop_sequence):
        self.trip_id = trip_id
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.stop_id = stop_id
        self.stop_sequence = stop_sequence

################################################################################


# <summary>
# Wrapper for the data in a GTFS file. Automatically loads the stop points and routes into internal structures ready for use by another class (maybe GTFSUtils?).
# This is a wrapper for a single file, not a whole directory of GTFS zip files like GTFSUtils handles.
# </summary>
class GTFSFile:

    # <summary>
    # Load a GTFS file. Filename must be a zip file.
    # </summary>
    # <param name="Filename"></param>
    def __init__(self, Filename):
        self.CurrentFilename = Filename
        self.Routes = {} #from routes.txt
        self.Stops = {} #from stops.txt
        self.Trips = {} #from trips.txt
        self.StopTimes = {} #from stop_times.txt

        with zipfile.ZipFile(Filename,'r') as zip:
            #load routes from routes.txt inside zip file
            with zip.open('routes.txt') as routestxt:
                self.ParseRoutes(routestxt)

            #load stop points from stops.txt inside zip file
            with zip.open('stops.txt') as stopstxt:
                self.ParseStops(stopstxt)

            #trips
            with zip.open('trips.txt') as tripstxt:
                self.ParseTrips(tripstxt)

            #stop_times
            with zip.open('stop_times.txt') as stoptimestxt:
                self.ParseStopTimes(stoptimestxt)
        #with ZipFile

################################################################################

        

    """
    Splits a string containing comma separated data and returns the elements as an array of strings.
    Quotes can be used around items containing a comma.
    These quotes are removed from the string array that is returned.
    
    <param name="line">The CSV line string to be split</param>
    <returns>Array of strings containing the comma separated elements in the CSV file line</returns>
    """
    def ParseCSVLine(self, line):
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
            elif ch=='"':
                Quote = not Quote
            else:
                Current += ch
            #end if ch==','
        #end for ch in line
        Items.append(Current.strip()) #add trailing item - even if last char was a comma, we still want a null on the end

        return Items

################################################################################


    # <summary>
    # routes.txt
    # NOTE: defaults to rail if no route_type found
    # </summary>
    # <param name="reader"></param>
    def ParseRoutes(self, reader):
        #route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color
        #1-BAK-_-y05-400122,OId_LUL,Bakerloo,,Elephant & Castle - Queen's Park - Harrow & Wealdstone,1,,,

        self.Routes = {}
        Line = reader.readline().decode('utf-8') #skip header
        Headers = self.ParseCSVLine(Line)
        HeaderLookup = {}
        for idx, Header in enumerate(Headers):
            HeaderLookup[Header]=idx

        for Line in reader:
            Line = Line.decode('utf-8') #this is nasty - the weakly typed language returns a byte array, which it then tries to parse as a string, including the leading b' meaning bytes!
            Fields = self.ParseCSVLine(Line)
            RouteId = Fields[HeaderLookup['route_id']]
            #AgencyId = Fields[HeaderLookup['agency_id']] #not in OASA data
            AgencyId = "OASA" #HACK!
            RouteShortName = Fields[HeaderLookup['route_short_name']]
            RouteLongName = Fields[HeaderLookup['route_long_name']]
            if 'route_desc' in HeaderLookup:
                RouteDesc = Fields[HeaderLookup['route_desc']]
            else:
                RouteDesc=''
            try:
                RouteType = int(Fields[HeaderLookup['route_type']])
            except ValueError as verr:
                RouteType = GTFSRouteType.Rail
            
            try:
                self.Routes[RouteId] = GTFSRoute(RouteId, AgencyId, RouteShortName, RouteLongName, RouteDesc, RouteType)
            except Exception as e:
                print('Error: routes.txt ' + self.CurrentFilename + ', ' + Line + ' ' + e)
        #end for

################################################################################


    # <summary>
    # reader points to a stops.txt file, which is read to extract stop locations and populate this.StopPoints.
    # </summary>
    # <param name="reader"></param>
    def ParseStops(self, reader):
        #stops.txt
        #stop_id,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url
        #9400ZZLUHAW1,"Wealdstone, Harrow & Wealdstone Station",,51.5923316813,-0.3354040827,,
        #9400ZZLUKEN2,"Kenton (London), Kenton",,51.5822158642,-0.3171684662,,

        #self.Stops = {}
        #Line = reader.readline() #skip header
        #for Line in reader:
        #    Line = Line.decode('utf-8')
        #    Fields = self.ParseCSVLine(Line)
        #    Code = Fields[0]
        #    Name = Fields[1]
        #    try:
        #        Lat = float(Fields[3])
        #        Lon = float(Fields[4])
        #        if not Code in self.Stops:
        #            self.Stops[Code] = GTFSStop(Code,Name,Lat, Lon)
        #    except Exception as e:
        #        print('Error: stops.txt ' + self.CurrentFilename + ', ' + str(Line) + ', ' + str(e))
        #    #end try except
        ##end for
        #
        #better code (python)
        dfStops = pd.read_csv(reader, dtype={'stop_id': str, 'stop_name': str})
        for index, row in dfStops.iterrows():
            #NOTE: MUST have the str() on the Code and Name for Python as they can use numbers! Weak typing mess!
            Code = row["stop_id"] #or stop_code?
            Name = row["stop_name"]
            try:
                Lat = float(row["stop_lat"])
                Lon = float(row["stop_lon"])
                if not Code in self.Stops:
                    self.Stops[Code] = GTFSStop(Code, Name, Lat, Lon)
            except Exception as e:
                Line = row[:].to_string(header=False, index=False)
                print('Error: stops.txt ' + self.CurrentFilename + ', ' + Line + ', ' + str(e))
            #end try except
        #end for iterrows
        print("ParseStops loaded "+str(len(self.Stops))+" stops")



################################################################################


    # <summary>
    # Reader points to a trips.txt file.
    # </summary>
    # <param name="reader"></param>
    def ParseTrips(self, reader):
        #route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id
        #3-E10-_-y05-41620,3-E10-_-y05-41620,VJ_3-E10-_-y05-41620-1-MF@05:20:00,Islip Manor Road - Ealing Broadway Station / Haven Green,0,,
        #3-E10-_-y05-41620,3-E10-_-y05-41620,VJ_3-E10-_-y05-41620-2-MF@05:22:00,Haven Green / Ealing Broadway - Islip Manor Road,1,,

        self.Trips = {}
        Line = reader.readline() #skip header
        for Line in reader:
            Line = Line.decode('utf-8')
            Fields = self.ParseCSVLine(Line)
            RouteId = Fields[0]
            ServiceId = Fields[1]
            TripId = Fields[2]
            TripHeadsign = Fields[3]
            try:
                if not TripId in self.Trips:
                    self.Trips[TripId] = GTFSTrip(RouteId,ServiceId,TripId,TripHeadsign)
            except Exception as e:
                print('Error: trips.txt ' + self.CurrentFilename + ', ' + Line + ', ' + e)
        #end for

################################################################################


    # <summary>
    # Parses a GTFS time code string, taking into account the fact that it might be of the form 24:49:00.
    # Arbitrarily sets all times to have a base date of 1 Jan 1970, so any hours>=24 will be set to the 2nd Jan.
    # </summary>
    # <param name="strTime"></param>
    # <returns></returns>
    def ParseGTFSTime(self, strTime):
        Fields = strTime.split(':')
        HH = int(Fields[0])
        MM = int(Fields[1])
        SS = int(Fields[2])
        Over24 = False
        if (HH >= 24):
            Over24 = True
            HH -= 24
        #end
        DT = datetime(1970,1,1,HH,MM,SS)
        if Over24:
             DT = DT + timedelta(hours=24)
        return DT

################################################################################


    def ParseStopTimes(self, reader):
        #trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
        #VJ_3-E10-_-y05-41620-1-MF@05:20:00,05:20:00,05:20:00,490008519N,1,,0,0,
        #VJ_3-E10-_-y05-41620-1-MF@05:20:00,05:20:00,05:20:00,490009557N,2,,0,0,

        self.StopTimes = {}
        Line = reader.readline() #skip header
        for Line in reader:
            Line = Line.decode('utf-8')
            Fields = self.ParseCSVLine(Line)
            try:
                TripId = Fields[0]
                #Annoyingly, you get some times recorded as 24:49:00, which is obviously over 24 hours. This is on services that run across midnight so you can tell it's not going backwards in time.
                #Need to filter this out though.
                #DateTime ArrivalTime = DateTime.Parse(Fields[1]); //DateTime - only the time is used, Date defaults to today
                #DateTime DepartureTime = DateTime.Parse(Fields[2]); //DateTime
                ArrivalTime = self.ParseGTFSTime(Fields[1])
                DepartureTime = self.ParseGTFSTime(Fields[2])
                StopId = Fields[3]
                StopSequence = int(Fields[4])

                if not TripId in self.StopTimes: #create a new TripId sequence
                    self.StopTimes[TripId] = []
                #end if
                #push new stop onto end of TripId sequence
                self.StopTimes[TripId].append(GTFSStopTime(TripId, ArrivalTime, DepartureTime, StopId, StopSequence))
            except Exception as e:
                print('Error: stop_times.txt ' + self.CurrentFilename + ', ' + Line + ', ' + e)
            #end try except
        #end for
    

################################################################################

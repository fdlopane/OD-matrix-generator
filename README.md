# Travel Cost Matrices Generator
### Description
This module generates a OD cost matrices (in minutes) for private (cars) and public transport.
Origins and destinations need to be provided as a point shp.
The example considers Oxford MSOA centroids as origins and destinations.
To run the code for a different city/region, change the input files (zones, centroids and GTFS)
and modify the "Case_Study_Area" variable.

### Required inputs:

Files:
* Origins: point shapefile (e.g. zones centroids)
* Zoning system polygons shapefile
* a .csv file containing information on the zoning system.
Note: the "zonei" column must contain the index of the zones starting at 1 and in ascending order.
This is fundamental as the public transport cost matrix will not have headers and will use the "zonei" index for rows
and columns.
* GTFS data for the case study area

Variables:
* Geographic definition of the case study area through the *Case_Study_Area* variable. The example provided is for the
city of Oxford. Just change the variable to whatever other city/region/country of interest. This will download the
driving road network from Open Street Maps.
* Walking node fixup distance through the variable *walkFixupDistMetres* (initialised by default at 500m).
This variable is used to connect network nodes that are walkable e.g. where there are two bus routes
but none of the stops are shared: a walking link between network segments is generated to make them connected.
(Note: recommended low values as it tends to add in lots of additional network links).

___
## Author:

[Dr Fulvio D. Lopane](https://fdlop.com/)

[The Bartlett Centre for Advanced Spatial Analysis](https://www.ucl.ac.uk/bartlett/casa/bartlett-centre-advanced-spatial-analysis)

[University College London](https://www.ucl.ac.uk/)
___
The public transport matrix generation is heavily based on [Richard Milton](https://www.ucl.ac.uk/bartlett/casa/mr-richard-milton)'s methodology also available in this [Github repository](https://github.com/maptube/Harmony)

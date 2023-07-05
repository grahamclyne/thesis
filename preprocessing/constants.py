CMIP_VARIABLES = {'Lmon':['cCwd','cVeg','cLitter','cLeaf','cRoot','evspsblsoi','lai','tsl','mrro','mrsos','grassFrac','shrubFrac','cropFrac','baresoilFrac','residualFrac','treeFrac','shrubFrac'],'Emon':['cSoilAbove1m','cSoil','cStem','wetlandFrac','cOther'],'Amon':['ps','pr','tas']}
RAW_CMIP_VARIABLES = ['cSoilAbove1m','cOther','cCwd','cVeg','cLitter','cLeaf','cRoot','evspsblsoi','lai','tsl','mrro','mrsos','grassFrac','shrubFrac','cropFrac','baresoilFrac','residualFrac','treeFrac','shrubFrac','cSoil','cStem','wetlandFrac','ps','pr','tas']
RAW_ERA_VARIABLES = ['2m_temperature', 'evaporation_from_bare_soil', 'leaf_area_index_high_vegetation','leaf_area_index_low_vegetation', 'runoff', 'skin_temperature','soil_temperature_level_1', 'surface_pressure', 'total_precipitation','volumetric_soil_water_layer_1']
PROCESSED_ERA_VARIABLES = ['t2m_DJF', 't2m_MAM', 't2m_JJA', 't2m_SON', 'year', 'lat', 'lon', 'evabs', 'lai_hv', 'lai_lv', 'ro', 'skt', 'stl1', 'sp', 'tp', 'swvl1']
RAW_NFIS_VARIABLES = ['no_change','water','snow_ice','rock_rubble','exposed_barren_land','bryoids','shrubs','wetland','wetland-treed','herbs','coniferous','broadleaf','mixedwood']
MODEL_INPUT_VARIABLES = ['evspsblsoi','lai','ps','tsl','mrro','mrsos','grassCropFrac','wetlandFrac','baresoilFrac','residualFrac','treeFrac','pr','tas_DJF', 'tas_MAM', 'tas_JJA', 'tas_SON',]
MODEL_TARGET_VARIABLES = ['cSoil','cCwd','cVeg','cLitter','cLeaf','cRoot','cStem','cOther']
# MODEL_TARGET_VARIABLES = ['cLeaf','cStem','cOther']
MODEL_INPUT_VARIABLES = ['lai','tsl','mrsos','grassCropFrac','wetlandFrac','baresoilFrac','residualFrac','treeFrac','tas_DJF', 'tas_MAM', 'tas_JJA', 'tas_SON',]


CMIP_EXPERIMENT = 'historical'
CMIP_VARIANT = 'r8i1p1f1'
CMIP_SOURCE = 'CESM2'
CMIP_NODE='aims3.llnl.gov'
"""
RAW ERA VARIABLES
MONTHLY DATA
t2m      2m temperature	K	Temperature of air at 2m above the surface of land, sea or in-land waters. 2m temperature is calculated by interpolating between the lowest model level and the Earth's surface, taking account of the atmospheric conditions. Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15.
evabs    Evaporation from bare soil	m of water equivalent	The amount of evaporation from bare soil at the top of the land surface. This variable is accumulated from the beginning of the forecast time to the end of the forecast step.
lai_hv   Leaf area index, high vegetation	m2 m-2	One-half of the total green leaf area per unit horizontal ground surface area for high vegetation type.
lai_lv   Leaf area index, low vegetation	m2 m-2	One-half of the total green leaf area per unit horizontal ground surface area for low vegetation type.
stl1     Soil temperature level 1	K	Temperature of the soil in layer 1 (0 - 7 cm) of the ECMWF Integrated Forecasting System. The surface is at 0 cm. Soil temperature is set at the middle of each layer, and heat transfer is calculated at the interfaces between them. It is assumed that there is no heat transfer out of the bottom of the lowest layer. Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15.
sp       Surface pressure	Pa	Pressure (force per unit area) of the atmosphere on the surface of land, sea and in-land water. It is a measure of the weight of all the air in a column vertically above the area of the Earth's surface represented at a fixed point. Surface pressure is often used in combination with temperature to calculate air density. The strong variation of pressure with altitude makes it difficult to see the low and high pressure systems over mountainous areas, so mean sea level pressure, rather than surface pressure, is normally used for this purpose. The units of this variable are Pascals (Pa). Surface pressure is often measured in hPa and sometimes is presented in the old units of millibars, mb (1 hPa = 1 mb = 100 Pa).
ro       Runoff	m	Some water from rainfall, melting snow, or deep in the soil, stays stored in the soil. Otherwise, the water drains away, either over the surface (surface runoff), or under the ground (sub-surface runoff) and the sum of these two is simply called 'runoff'. This variable is the total amount of water accumulated from the beginning of the forecast time to the end of the forecast step. The units of runoff are depth in metres. This is the depth the water would have if it were spread evenly over the grid box. Care should be taken when comparing model variables with observations, because observations are often local to a particular point rather than averaged over a grid square area. Observations are also often taken in different units, such as mm/day, rather than the accumulated metres produced here. Runoff is a measure of the availability of water in the soil, and can, for example, be used as an indicator of drought or flood. More information about how runoff is calculated is given in the IFS Physical Processes documentation.
tp       Total precipitation	m	Accumulated liquid and frozen water, including rain and snow, that falls to the Earth's surface. It is the sum of large-scale precipitation (that precipitation which is generated by large-scale weather patterns, such as troughs and cold fronts) and convective precipitation (generated by convection which occurs when air at lower levels in the atmosphere is warmer and less dense than the air above, so it rises). Precipitation variables do not include fog, dew or the precipitation that evaporates in the atmosphere before it lands at the surface of the Earth. This variable is accumulated from the beginning of the forecast time to the end of the forecast step. The units of precipitation are depth in metres. It is the depth the water would have if it were spread evenly over the grid box. Care should be taken when comparing model variables with observations, because observations are often local to a particular point in space and time, rather than representing averages over a model grid box and model time step.
swvl1    Volumetric soil water layer 1	m3 m-3	Volume of water in soil layer 1 (0 - 7 cm) of the ECMWF Integrated Forecasting System. The surface is at 0 cm. The volumetric soil water is associated with the soil texture (or classification), soil depth, and the underlying groundwater level. m3m-3 is a ratio
"""

"""
RAW NFIS VARIABLES
0 = no change
20 = water
31 = snow_ice
32 = rock_rubble
33 = exposed_barren_land
40 = bryoids
50 = shrubs
80 = wetland
81 = wetland-treed
100 = herbs
210 = coniferous
220 = broadleaf
230 = mixedwood
"""


"""
RAW CESM VARIABLES

tas             Near-Surface Air Temperature	K	near-surface (usually, 2 meter) air temperature   
evspsblsoi      Water Evaporation from Soil	kg m-2 s-1	Water evaporation from soil (including sublimation).   
lai             Leaf Area Index	1	A ratio obtained by dividing the total upper leaf surface area of vegetation by the (horizontal) surface area of the land on which it grows.    
tsl             Temperature of Soil	K	Temperature of soil. Reported as missing for grid cells with no land.    
ps              Surface Air Pressure	Pa	surface pressure (not mean sea-level pressure), 2-D field to calculate the 3-D pressure field from hybrid coordinates		
mrro            Total Runoff	kg m-2 s-1	The total run-off (including drainage through the base of the soil model) per unit area leaving the land portion of the grid cell.    
pr              Precipitation	kg m-2 s-1	includes both liquid and solid phases	at surface; includes both liquid and solid phases from all types of clouds (both large-scale and convective)
mrsos           Moisture in Upper Portion of Soil Column	kg m-2	The mass of water in all phases in the upper 10cm of the  soil layer.	the mass of water in all phases in a thin surface soil layer.


*** TODO: ADD CARBON TARGET VAIRABLES ****
Natural Grass Area Percentage	%	Percentage of entire grid cell that is covered by natural grass.		grassFrac
Percentage Cover by Shrub	%	Percentage of entire grid cell  that is covered by shrub.	fraction of entire grid cell  that is covered by shrub.	shrubFrac
Percentage Crop Cover	%	Percentage of entire grid cell  that is covered by crop.		cropFrac
Bare Soil Percentage Area Coverage	%	Percentage of entire grid cell  that is covered by bare soil.		baresoilFrac
Percentage of Grid Cell That Is Land but neither Vegetation Covered nor Bare Soil	%	Percentage of entire grid cell  that is land and is covered by  neither vegetation nor bare-soil (e.g., urban, ice, lakes, etc.)	fraction of entire grid cell  that is land and is covered by "non-vegetation" and "non-bare-soil" (e.g., urban, ice, lakes, etc.)	residualFrac
Tree Cover Percentage	%	Percentage of entire grid cell  that is covered by trees.	fraction of entire grid cell  that is covered by trees.	treeFrac
"""



"""
CONVERT ERA TO CESM (30.437 avg number of days in a month)
t2m -> tas 
(evabs * 1000)/ ((24*60*60*30.437) -> evspsblsoi  #1000 here bc 1kg/m^2 is 1mm of thickness of water
lai_hv + lai_lv -> lai
stl1 -> tsl
sp -> ps
(ro * 1000)/ ((24*60*60*30.437)  -> mrro
(tp * 1000)/ ((24*60*60*30.437)  -> pr
swvl1 * 100 -> mrsos 


The term volumetric water content (VWC) refers to a given volume of water contained in
 given volume of substrate (soil or any other soiless media that can soak water). 
 this makes the term have a unit of measurement m3/m3. that means a given m3 of 
 water contained in a given m3 of soil or other media. In short the units (m3/m3) 
 cancels out virtually. So if your device measure for instance 20 % VWC in a soil 
 depth of say 40 cm, it means, of the 40 cm depth in the soil there contains 20 % water. 
 Hence the water content in terms of depth (cm),i.e. 40 cm soil depth is 20% of 40 cm=20/100 * 40 cm= 8 cm.
  if you wanted to have your final answer in mm then you convert 8 cm to mm which you multiply
   by 10 as the conversion factor from cm to mm. Your final answer therefore = 80 mm. 
   this 80 mm you have as your final answer means, when you dig 40 cm deep in the earth of
    soil you will have 80 mm of water in that soil. I hope it is clear.

CONVERT NFIS TO CESM 

wetland -> wetlandFrac
coniferous + broadleaf + mixedwood + wetland-treed -> treeFrac
herbs + bryoids-> grassFrac + cropFrac (this means CESM way overestimates grass/crop fracs)
exposed barren land -> baresoilFrac (nfis is larger estimation of this)
rock-rubble + snow-ice + water -> residualFrac
"""







"""
MODEL INPUT VARIABLES (INDEPENDENT VARIABLES)
2m temp
tree frac 

"""


"""
MODEL TARGET VARIABLES (DEPENDENT VARIABLES)


"""

import datetime as dt
from ravenpy.models import GR4JCN, HMETS, MOHYSE, HBVEC, Raven
from ravenpy.utilities.testdata import get_file
from matplotlib import pyplot as plt
model = HBVEC()
files = ['raven-hbv-ec/raven-hbv-ec-salmon.rvt', "raven-hbv-ec/raven-hbv-ec-salmon.rvc",
"raven-hbv-ec/raven-hbv-ec-salmon.rvi","raven-hbv-ec/raven-hbv-ec-salmon.rvh",
"raven-hbv-ec/raven-hbv-ec-salmon.rvp"]#"raven-hbv-ec/Salmon-River-Near-Prince-George_meteo_daily.rvt"]
config = [
    get_file(i) for i in files
]

forcing = get_file(
    "raven-gr4j-cemaneige/Salmon-River-Near-Prince-George_meteo_daily.nc"
   # "raven-hbv-ec/Salmon-River-Near-Prince-George_meteo_daily.rvt"
)

model(
    forcing,
    start_date=dt.datetime(2000, 1, 1),
    end_date=dt.datetime(2002, 1, 1),
    area=4250.6,
    elevation=843.0,
    latitude=54.4848,
    longitude=-123.3659,
    tas=[],  
    params=(0.1, 0.1, 0.29, 1.0, .9, 
    0.947,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,0.1)
)
model.configure(config)

# model(forcing)
print(model.hydrograph)

model.q_sim.plot()
plt.show()
import requests
import requests



lat = 61.3089
lon = -121.2984

for i in range(100,300,8):
    #latitude, longitude, product, startDate, endDate, kmAboveBelow, kmLeftRight
    start_date = 'A2021' + str(i)
    end_date = 'A2021' + str(i + 8)
    print(start_date, end_date)
    query = {'latitude':lat, 'longitude':lon, 'product':'MOD11A2', 'startDate':start_date, 'endDate':end_date, 'kmAboveBelow':1, 'kmLeftRight':1}
    response = requests.get("https://modis.ornl.gov/rst/api/v1/MOD11A2/subset", params=query)

    data = {}
    print(response)
    print(response.json())
    subset = response.json()['subset']
    for i in subset:
        if(i['band'] == 'LST_Day_1km'):
            data[i['modis_date']] = i['data']
print(data)
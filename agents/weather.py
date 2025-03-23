from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import json
from openmeteo_sdk.Variable import Variable
import requests
from pathlib import Path  
import os
from dotenv import load_dotenv

env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

azuremapssubskey = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")
azuremapsclientid = os.getenv("AZURE_MAPS_CLIENT_ID")


def get_weather_info(location):

    address=location
    geolocator = Nominatim(user_agent="Dummy")
    location = geolocator.geocode(address)
    # print(location.address)
    # print((location.latitude, location.longitude))

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": "temperature_2m",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process current data. The order of variables needs to be the same as requested.
    current = response.Current()
    current_variables = list(map(lambda i: current.Variables(i), range(0, current.VariablesLength())))
    current_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, current_variables))
    # print(f"Current temperature_2m {current_temperature_2m.Value()}")   


    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    # print(hourly_temperature_2m)

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_dataframe = pd.DataFrame(data = hourly_data)

    # Convert to JSON
    json_string = hourly_dataframe.to_json(orient='records')
    # print("Current tempeature at ",  address, " :: ", json.loads(json_string)[0].get("temperature_2m"))

    # inttemp = int(json.loads(json_string)[0].get("temperature_2m"))
    inttemp = int(current_temperature_2m.Value())
    # print("Decimal Temp :: ", inttemp)

    return "Current tempeature at ",  location, " :: ", inttemp, "Degree Fahrenheit"


def get_weather_from_azure_maps(location):

    address=location
    geolocator = Nominatim(user_agent="Dummy")
    location = geolocator.geocode(address)
    lat = str(location.latitude)
    lon = str(location.longitude)

    latlon = lat + "," + lon

    url = "https://atlas.microsoft.com/weather/currentConditions/json?api-version=1.0&query=" + latlon + "&subscription-key=" + azuremapssubskey

    headers = {
        'Content-Type': 'application/json',
        'x-ms-client-id': azuremapsclientid        
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response Body:", response.json())
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# get_weather_info("Atlanta")
# get_weather_from_azure_maps("Atlanta")
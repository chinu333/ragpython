import requests
import urllib.parse as urlparse
import datetime
from geopy.geocoders import Nominatim
from pathlib import Path  
import os
from dotenv import load_dotenv


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

tomtomapikey = os.getenv("TOMTOM_MAPS_API_KEY")


def get_traffic_info(start_address, end_address):
 
    # Route parameters
    slat, slong = get_coordinates(start_address)
    elat, elong = get_coordinates(end_address)
    # start = "37.77493,-122.419415"                                              # San Francisco
    # end = "34.052234,-118.243685"                                               # Los Angeles
    start = str(slat) + ',' + str(slong)                                          
    end = str(elat) + ',' + str(elong) 
    routeType = "fastest"                                                         # Fastest route
    traffic = "true"                                                              # To include Traffic information
    travelMode = "car"                                                            # Travel by car
    avoid = "unpavedRoads"
    # departAt = "2021-10-20T10:00:00"                                            # Avoid unpaved roads
    departAt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")              # Departure date and time
    vehicleCommercial = "false"                                                    # Commercial vehicle
    key = tomtomapikey                                      # API Key
    
    # Building the request URL
    baseUrl = "https://api.tomtom.com/routing/1/calculateRoute/"
    
    requestParams = (
        urlparse.quote(start) + ":" + urlparse.quote(end) 
        + "/json?routeType=" + routeType
        + "&traffic=" + traffic
        + "&travelMode=" + travelMode
        + "&avoid=" + avoid 
        + "&vehicleCommercial=" + vehicleCommercial
        + "&departAt=" + urlparse.quote(departAt))
    
    requestUrl = baseUrl + requestParams + "&key=" + key

    response = requests.get(requestUrl)

    if(response.status_code == 200):
        # Get response's JSON
        jsonResult = response.json()
    
        # Read summary of the first route
        routeSummary = jsonResult['routes'][0]['summary']
        print(routeSummary)
        
        # Read ETA
        eta = routeSummary['arrivalTime']
    
        # Read travel time and convert it to hours
        travelTime = routeSummary['travelTimeInSeconds'] / 3600

        travelTimeMins = routeSummary['travelTimeInSeconds'] / 60
        
        # Print results
        print(f"{departAt}, ETA: {eta}, Travel time: {travelTime:.2f}h, Travel time in mins: {travelTimeMins:.2f}m")

        return routeSummary

def get_coordinates(address):
    geolocator = Nominatim(user_agent="Dummy")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude

# home = "1800 Greystone Summit Drive, Cumming, GA, 30040"
# work = "200 17th St NW, Atlanta, GA, 30363"

# home = "Cumming, GA, 30040"
# work = "Atlanta, GA, 30363"

# print(get_traffic_info(home, work))
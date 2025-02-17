import requests
import json


def get_nutrition_info(query):

    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    payload = {
        "query": query
    }

    headers = {
        'Content-Type': 'application/json',
        'x-app-id': 'aaaa',
        'x-app-key': 'bbbb'  
        
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        # print("Status Code:", response.status_code)
        # print("Response Body:", response.json())
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# get_nutrition_info("1 cup of rice")
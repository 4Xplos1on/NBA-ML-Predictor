import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("Testing API Key...")

headers = {
    # Ensure there are absolutely NO spaces before or after the key inside these quotes
    "Authorization": "cf0dff1a-8ac4-469f-9071-9e5c5515880f" 
}

url = "https://api.balldontlie.io/v1/stats?seasons[]=2023&player_ids[]=237"

response = requests.get(url, headers=headers, verify=False)

if response.status_code == 200:
    print("Success! The key works.")
    data = response.json()['data']
    df = pd.DataFrame([game['stat'] for game in data])
    print(df[['pts', 'ast', 'reb']].head())
else:
    print(f"Failed to connect. Server said: {response.status_code}")
    # This new line will print the exact reason the server rejected your key
    print(f"Reason from server: {response.text}")

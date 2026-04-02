import requests
import pandas as pd
import urllib3

# Disable warning for verify false
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Test API key
print("Testing API Key")

# API key header
headers = {
    "Authorization": "cf0dff1a-8ac4-469f-9071-9e5c5515880f"
}

# Request URL
url = "https://api.balldontlie.io/v1/stats?seasons[]=2023&player_ids[]=237"

# Send request
response = requests.get(url, headers=headers, verify=False)

if response.status_code == 200:
    print("Success")
    
    # Get data
    data = response.json()['data']
    df = pd.DataFrame([game['stat'] for game in data])
    
    # Show sample
    print(df[['pts', 'ast', 'reb']].head())
else:
    print("Failed to connect")
    
    # Show status and message
    print(response.status_code)
    print(response.text)

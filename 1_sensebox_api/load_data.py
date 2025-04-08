import requests
import json
import pandas as pd

SENSEBOX_ID = "5ea96b86cc50b1001b78fe27"

class SenseBox():
    def __init__(self, sensebox_id=SENSEBOX_ID):
        self.sensebox_id=sensebox_id
        self.url = "https://api.opensensemap.org"

        response = requests.request(method='get', 
                                    url=f"{self.url}/boxes/{self.sensebox_id}", 
                                    params={"resaponse_format": "json"})
        if response.status_code != 200:
            print(response.raise_for_status())

        self.box_info = response.json()

        self.known_sensors = self.box_info['sensors']
        for sensor in self.known_sensors:
            sensor['lastMeasurement'] = sensor['lastMeasurement']['createdAt']

    def box_information(self, print_information=False):
        df = pd.json_normalize(self.box_info)
        print(df.transpose())
    
    def sensor_information(self):
        df = pd.DataFrame(self.known_sensors)
        print(df)
        

    def get_sensor_data(self, sensor_id, print_information=False):
        #https://api.opensensemap.org/boxes/:senseBoxId/data/:sensorId?from-date=fromDate&to-date=toDate&download=true&format=json
        response = requests.request(method='get', 
                            url=f"{self.url}/boxes/{self.sensebox_id}/data/{sensor_id}", 
                            params={"resaponse_format": "json"})
        if response.status_code != 200:
            print(response.raise_for_status())

        data = response.json()
        if print_information:
            df = pd.json_normalize(data)
            print(df)
        return(data)


if __name__ == "__main__":
    sb = SenseBox() #.get_sensor_data("6730f1885f78e9000736a078")
    sb.sensor_information()
    sb.get_sensor_data("6730f1885f78e9000736a078", print_information=True)

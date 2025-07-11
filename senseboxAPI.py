"""SenseBox: API class for sense boxes"""

import requests
import pandas as pd
from copy import deepcopy
import datetime as dt
from datetime import timezone
import warnings

class SenseBox():
    """Connection to one SenseBox and allows to get the Information
    about the box, the sensors and the data collected by the sensors.\n
    sb = Sensebox(senseboxID)\n
    sb.get_box_info()               #general info\n
    sb.get_sensor_info()            #sensorId, \n
    sb.get_sensor_data(SensorID)    #measured values\n\n
    sb.request_box_info()           #renew manually the box info
    """
    def __init__(self, sensebox_id: str):
        """Connection to one SenseBox
        Args:
            sensebox_id (str): Sensbox Id from https://opensensemap.org"""
        self.sensebox_id: str = sensebox_id
        self.url: str = 'https://api.opensensemap.org'
        self.request_box_info()

    def request_box_info(self) -> None:
        """Updates internaly the box- and sensor information.
        """
        response = requests.request(method='get', 
                                    url=f'{self.url}/boxes/{self.sensebox_id}', 
                                    params={'resaponse_format': 'json'})
        if response.status_code != 200:
            print(response.raise_for_status())
        self._box_info = response.json()

        self.known_sensors: list[dict] = self._box_info['sensors']
        for sensor in self.known_sensors:
            sensor['lastMeasurement'] = sensor['lastMeasurement']['createdAt']

    def get_box_info(self) -> dict:
        """Returns the basic Informationn about the box as dictionary:
        keys = ['box_id', 'createdAt', 'longitude' and 'latitude']
        """
        return_dict = {'box_id': self._box_info['_id'],
                       'createdAt': dt.datetime.fromisoformat(self._box_info['createdAt'].replace('Z', '+00:00')),
                       'longitude': self._box_info['currentLocation']['coordinates'][0],
                       'latitude': self._box_info['currentLocation']['coordinates'][0],
                       }
        return return_dict
    
    def get_sensor_info(self) -> list[dict]:
        """Returns a dictionary with information about each sensors in a list:\n
        keys = ['title', 'unit', 'sensorType', 'icon', 'lastMeasurement', 'sensor_id']\n
        Timezone in UTC
        """
        sensor_list = deepcopy(self.known_sensors)
        for sensor_dict in sensor_list:
            sensor_dict['sensor_id'] = sensor_dict.pop('_id')
            sensor_dict['lastMeasurement'] = \
                dt.datetime.fromisoformat(sensor_dict['lastMeasurement'].replace('Z', '+00:00')) #timezone: UTC
        return sensor_list

    def __get_sensor_data_batch(self, sensor_id: str,\
                        datetime_from: dt.datetime=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=2), \
                        datetime_to: dt.datetime=dt.datetime.now(tz=timezone.utc)):
        """Returns the measured values from a specific sensor.\n
        Maximum time between datetime_from and datetime_to is 30 days. Each has to be in UTC.
        """
        def __sensor_single_batch(sensor_id: str, datetime_from: dt.datetime, datetime_to: dt.datetime, iteration: int=0):
            print("\r" + f"Request batch {self.batch_number}/{self.batch_len} with iteration: {iteration}", end="")
            if datetime_from < datetime_to:
                datetime_from_str: str = datetime_from.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                datetime_to_str: str = datetime_to.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

                params={'resaponse_format': 'json',
                        'from-date': datetime_from_str,
                        'to-date': datetime_to_str,
                        }
                response = requests.request(method='get',
                                            url=f'{self.url}/boxes/{self.sensebox_id}/data/{sensor_id}', 
                                            params=params)
                if response.status_code != 200:
                    print(response.raise_for_status())
                data = response.json()
                if iteration > 3:
                     warnings.warn('Too many datapoints in one interval -> recursion is going to deep.\n \
                                   Some datapoins will be missing ', RuntimeWarning)
                     return data
                if len(data) >= 10000:
                    new_interval_length = (datetime_to - datetime_from) /2
                    new_time_interval = [[datetime_from, datetime_from + new_interval_length], 
                                          [datetime_from + new_interval_length + dt.timedelta(seconds=0.001), datetime_to]]
                    data = __sensor_single_batch(sensor_id, new_time_interval[0][0], new_time_interval[0][1], iteration=iteration+1)
                    self.batch_number -= 1
                    data += __sensor_single_batch(sensor_id, new_time_interval[1][0], new_time_interval[1][1], iteration+1)
                    self.batch_number -= 1
            else:
                data = []
            self.batch_number += 1
            return data
        
        data = __sensor_single_batch(sensor_id, datetime_from, datetime_to)
        return data

    def get_sensor_data(self, sensor_id: str,\
                        datetime_from: dt.datetime=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=2), \
                        datetime_to: dt.datetime=dt.datetime.now(tz=timezone.utc))-> pd.DataFrame | None:
        """Returns the measured values from a specific sensor.
        Args: 
            datetime_from (datetime.datetime): First including datetime
            datetime_to (datetime.datetime): Last including datetime
            
            datetime_from and datetime_to  has to be in UTC (tz=timezone.utc).
        """
        assert datetime_from.tzname() == 'UTC', 'datetime_from has to be utc'
        assert datetime_to.tzname() == 'UTC', 'datetime_to has to be utc'
        
        last_sensor_activity = dt.datetime.now(tz=timezone.utc)
        for sensor in self.known_sensors:
            if sensor['_id'] == sensor_id:
                last_sensor_activity = dt.datetime.fromisoformat(sensor['lastMeasurement'].replace('Z', '+00:00'))
        if datetime_to < last_sensor_activity:
            to_date = datetime_to
        else:
            to_date = last_sensor_activity
        self.batch_number = 1
        self.batch_len = 1
        from_date = datetime_from  # first possilbe date: sensor initialisation not implemented (movied to dbm)
        to_date = datetime_to  # last possible date: last sensor measuremet not implemented (moved to dbm)
        if to_date - from_date > dt.timedelta(days=1):
            datapoints_per_day = len( self.__get_sensor_data_batch(sensor_id, \
                                                            dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=1), \
                                                            dt.datetime.now(tz=timezone.utc)))
            if datapoints_per_day == 0:
                datapoints_per_day = 1
            days_for_10k_datapoints = 10000 / datapoints_per_day
            if days_for_10k_datapoints < 30:
                request_interval_len = dt.timedelta(days=days_for_10k_datapoints * 0.98)
            else:
                request_interval_len = dt.timedelta(days=30)

            time_intervals: list[list[dt.datetime]] = [[from_date, from_date + request_interval_len - dt.timedelta(seconds=0.001)]]
            while time_intervals[-1][1] < to_date:
                time_intervals.append([time_intervals[-1][1] + dt.timedelta(seconds=0.001)\
                                       , time_intervals[-1][1] + request_interval_len + dt.timedelta(seconds=0.001)])
            time_intervals[-1][1] = to_date

            data = []
            self.batch_len = len(time_intervals) + 1
            for interval in time_intervals:
                data += self.__get_sensor_data_batch(sensor_id, interval[0], interval[1])
        else:
            data = self.__get_sensor_data_batch(sensor_id, from_date, to_date)
        print('')
        data = pd.DataFrame(data)
        if len(data) > 0:
            data["measurement"] = pd.to_numeric(data["value"], errors="coerce")
            data["measurement_time"] = pd.to_datetime(data["createdAt"].str.replace('Z', '+00:00'))
            return data[["measurement_time", "measurement"]]
        else:
            return None

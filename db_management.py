import pandas as pd
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError
import datetime as dt
from datetime import timezone
from senseboxAPI import SenseBox


class DBManagement():
    """"Managing the Database and request from the sensbox. The DB can save data from multiple senseboxes, 
    but will write/read for the connected one."""
    def __init__(self, boxId: str):
        """Managing the Database and request from the connected sensbox with boxId
        Args:
            boxId (str): Sensebox ID for the connected"""
        self.boxId: str = boxId
        self.sb = SenseBox(self.boxId)
    
    def db_setup(self, default_ports=True):
        """"Basic configuration for the DB"""
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sensor_data (
                        box_id              VARCHAR(24),
                        sensor_id           VARCHAR(24),
                        measurement_time    TIMESTAMPTZ,
                        measurement         DOUBLE PRECISION,
                        unit                VARCHAR(16),
                        sensor_type         VARCHAR(50),
                        title               VARCHAR(50),
                        icon                VARCHAR(50),
                        PRIMARY KEY (sensor_id, measurement_time)
                    );
                """))

        except SQLAlchemyError as error:
            print(error)
    
    def db_reset(self):
        """Resets the database (deleting the table with data)"""
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE sensor_data;"))
        self.db_setup()


    def write_new_data(self, datetime_from=None, datetime_to: None | dt.datetime=None,):
        """Requests all new data from all sensors from the connected sensebox.
        Args:
            datetime_from (datetime.datetime): first included datetime. If None, date of box creation is used.
            datetime_to (datetime.datetime): last included datetime. If None, last date from SenseBox.request_box_info() is used.
        """
        self.sb.request_box_info()
        sensor_info = self.sb.get_sensor_info()
        for sensor in sensor_info:
            sensor_id = sensor["sensor_id"]

            if type(datetime_to) != dt.datetime:
                datetime_to = sensor["lastMeasurement"]
            assert type(datetime_to) == dt.datetime

            last_db_datapoint = self._last_saved_measurement_db(sensor_id)
            print(datetime_from)
            if type(datetime_from) == int:
                datetime_from = datetime_to - dt.timedelta(days=datetime_from)
            if type(datetime_from) != dt.datetime:
                datetime_from = self.sb.get_box_info()["createdAt"]
            if type(last_db_datapoint) == dt.datetime:
                if datetime_from < last_db_datapoint:
                    datetime_from = last_db_datapoint + dt.timedelta(seconds=1)
            assert type(datetime_from) == dt.datetime

            print(f"sensor_id: {sensor_id}; title: {sensor["title"]}; datetime_from: {datetime_from}; datetime_to: {datetime_to}")
            if datetime_from <= datetime_to:
                data = self.sb.get_sensor_data(sensor_id, 
                                               datetime_from=datetime_from,
                                               datetime_to=datetime_to)
                if type(data) == pd.DataFrame:
                    for column in ["unit", "title", "icon",]:
                        data[column] = sensor[column]
                    data["sensor_type"] = sensor["sensorType"]
                    data["sensor_id"] = sensor_id
                    data["box_id"] = self.sb.sensebox_id
                    print(f"{len(data)} datapoints")

                    engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
                    data.to_sql('sensor_data', con=engine, if_exists='append', index=False)
                else:
                    print("No datapoints available")
            else:
                print("datetime_from is after then datetime_to")

    def _get_db_sensorIds(self) -> list[str]:
        """Returns all sensors in the database from the connected sensebox
        """
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
        query = text(f"SELECT DISTINCT sensor_id FROM sensor_data where box_id='{self.boxId};")

        results = []
        with engine.connect() as connection:
            result_proxy = connection.execute(query)
            for row in result_proxy:
                results.append(row[0])  # Assuming you're interested in the first column
        return results

    def _last_saved_measurement_db(self, sensor_id) -> dt.datetime | None:
        """Returns the most recent timestamp of a given sensor in the database
        Args:
            sensor_id (str): SensorID of the Sensor
            """
    
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
        query = text(f"""SELECT MAX(measurement_time) from sensor_data
                WHERE sensor_id = '{sensor_id}'
                ;""")
        with engine.begin() as conn:
            newest = conn.execute(query)
            max_time = newest.scalar()
        return max_time
    
    def read_data(self)-> pd.DataFrame:
        """Reads the data of the connected sensebox from the database
        """
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/env_monitoring")
        query = f"SELECT * from sensor_data where box_id='{self.boxId}'"
        df = pd.read_sql(query, con=engine)
        return df

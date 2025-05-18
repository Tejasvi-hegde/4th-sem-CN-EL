from influxdb_client import InfluxDBClient

class InfluxDBWriter:
    def __init__(self):
        self.client = InfluxDBClient(url="http://localhost:8086", token="JU8UDuL3viuNAVfv6tRI_HpCXH-oO4RaF5JyOAYv7qzGuOpPNt8EWSy5blc5YuCYwEjxAI50rotWYErz2LzjyA==",org="network_monitor")
        self.write_api = self.client.write_api()
    
    def write(self, metrics):
        self.write_api.write(
            bucket="cca_metrics",
            record={
                "measurement": "network_metrics",
                "fields": metrics
            }
        )
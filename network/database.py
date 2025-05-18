from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

import subprocess
import time
import monitor as m
class InfluxDBWriter:
    def __init__(self):
        self.client = InfluxDBClient(
            url="http://localhost:8086",
            token="JU8UDuL3viuNAVfv6tRI_HpCXH-oO4RaF5JyOAYv7qzGuOpPNt8EWSy5blc5YuCYwEjxAI50rotWYErz2LzjyA==",
            org="network_monitor"
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        

    # Add this to database.py
    def test_connection(self):
        try:
            # Try to query buckets to test connection
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets().buckets
            print("Connected to InfluxDB. Available buckets:")
            for bucket in buckets:
                print(f"- {bucket.name}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    
    def get_current_cca(self):
        try:
            result = subprocess.run(
                "ip netns exec ns-client sysctl net.ipv4.tcp_congestion_control",
                shell=True, capture_output=True, text=True
            )
            return result.stdout.split('=')[-1].strip()
        except Exception as e:
            print(f"Error getting CCA: {e}")
            return "unknown"
    
    def write_metrics(self, metrics):
        start_time = time.time()
        print(f"Preparing to write metrics: {metrics}") 
        data_point = {
            "measurement": "network_metrics",
            "tags": {
                "host": "ns-client",
                "cca": metrics.get('current_cca', 'unknown')
            },
            "fields": {
                "rtt": float(metrics['rtt']),
                "throughput": float(metrics['throughput']),
                "loss": float(metrics['loss']),
                "bdp": float(metrics['bdp']),
                "bufferbloat": float(metrics['bufferbloat']),
                "retransmits": int(metrics['retransmits'])
            }
        }
        print(f"Write latency: {time.time() - start_time:.2f}s")
        try:
            self.write_api.write(
                bucket="cca_metrics",
                org="network_monitor",
                record=data_point
            )
            print("Successfully wrote metrics to InfluxDB")  # Debug line
        except Exception as e:
            print(f"Error writing to InfluxDB: {e}")
    
    def collect_metrics(self):
        metrics = {
        'rtt': m.get_rtt(),
        'throughput': m.get_throughput(),
        'loss': m.get_loss(),
        'bdp': m.calculate_bdp(),
        'bufferbloat': m.measure_bufferbloat(),
        'retransmits': m.get_retransmits()['retransmits'],
        'current_cca': self.get_current_cca()
    }
        return metrics
    
    def __del__(self):
        self.client.close()



# Initialize writer
#influx_writer = InfluxDBWriter()

# Periodic collection
#while True:
    
 #   metrics = influx_writer.collect_metrics()
  #  influx_writer.write_metrics(metrics)
  #  time.sleep(10)
#from monitor import NetworkMonitor
#monitor = NetworkMonitor()
#print(monitor.collect_all_metrics())
#from decision_engine import DecisionEngine
#engine = DecisionEngine(use_ml=False)
#engine.decide()  # Should use only rule-based logic
#import sys
#sys.path.append("/home/tejasvi/Desktop/CN EL 4th sem/")
#from ml.data_collector import CCADataCollector
#collector = CCADataCollector()
#collector.generate_training_data()
# main.py
from monitor import NetworkMonitor
import time

monitor = NetworkMonitor()
while True:
    monitor.collect_all_metrics()
    time.sleep(10)  # Collect every 10 seconds
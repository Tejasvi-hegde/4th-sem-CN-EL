import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU if not needed
import time
import numpy as np
import sys
sys.path.append("/home/tejasvi/Desktop/CN EL 4th sem/")
from ml.data_collector import CCADataCollector
from ml.inference import CCASelector
from ml.train_rf import CCAModelTrainer
from network.monitor import NetworkMonitor
from network.decision_engine import DecisionEngine

class MLCCAOptimizer:
    def __init__(self, model_type='rf'):
        self.collector = CCADataCollector()
        self.selector = CCASelector(model_type=model_type)
        self.current_cca = 'cubic'
        self.monitor = NetworkMonitor()
        self.decision_engine = DecisionEngine() 
        self._init_model()

    def _init_model(self):
        """Ensure model exists or train initial model"""
        try:
            self.selector.reload_model()
        except FileNotFoundError:
            print("No model found. Training initial model...")
            self.collector.generate_training_data()
            CCAModelTrainer().train('data/cca_training_data.parquet')
            self.selector.reload_model()

    def _get_metrics(self):
        """Collect metrics with validation"""
        metrics = {
            'rtt': self.monitor.get_rtt(),
            'throughput': self.monitor.get_throughput(),
            'loss': self.monitor.get_loss(),
            'retransmits': self.monitor.get_retransmits(),
            'timestamp': time.time()
        }
        
        # Validate metrics
        if any(v is None for v in metrics.values()):
            raise ValueError("Missing metric values")
        
        return metrics

    def _should_retrain(self):
        """Check retraining conditions"""
        # Weekly retraining
        if time.time() % 604800 < 5:  
            return True
            
        # Emergency retrain if performance drops
        performance = self.collector.calculate_performance()
        if performance < 0.7:  # 70% of target throughput
            return True
            
        return False

    def run(self):
        """Main optimization loop"""
        try:
            print("Starting MLCCAOptimizer...")
            while True:
                try:
                    print("Collecting metrics...")
                    # Collect and validate metrics
                    metrics = self._get_metrics()
                    print(f"Current metrics: {metrics}")
                    # Make prediction
                    predicted_cca = self.selector.predict(metrics)
                    
                    # Switch CCA if needed
                    if predicted_cca != self.current_cca:
                        print(f"Switching CCA to: {predicted_cca}")
                        self.decision_engine._switch_cca(predicted_cca)
                        self.current_cca = predicted_cca
                        self.collector.log_switch(predicted_cca, metrics)
                    
                    # Periodic retraining
                    if self._should_retrain():
                        print("Retraining model...")
                        self.collector.generate_training_data()
                        CCAModelTrainer().train('data/cca_training_data.parquet')
                        self.selector.reload_model()
                    print(f"Sleeping for 10 seconds...")
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            print("Gracefully shutting down...")
            self.collector.cleanup()
# Add at the end of mainml.py
if __name__ == "__main__":
    optimizer = MLCCAOptimizer()
    optimizer.run()
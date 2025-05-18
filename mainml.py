import time
import numpy as np
from ml.data_collector import CCADataCollector
from ml.inference import CCASelector
from ml.models.train_rf import CCAModelTrainer
from network.monitor import get_rtt, get_throughput, get_loss
from network.decision_engine import set_cca

class MLCCAOptimizer:
    def __init__(self, model_type='rf'):
        self.collector = CCADataCollector()
        self.selector = CCASelector(model_type=model_type)
        self.current_cca = 'cubic'
        self._init_model()

    def _init_model(self):
        """Ensure model exists or train initial model"""
        try:
            self.selector.load_model()
        except FileNotFoundError:
            print("No model found. Training initial model...")
            self.collector.generate_training_data()
            CCAModelTrainer().train('data/cca_training_data.parquet')
            self.selector.load_model()

    def _get_metrics(self):
        """Collect metrics with validation"""
        metrics = {
            'rtt': get_rtt(),
            'throughput': get_throughput(),
            'loss': get_loss(),
            'retransmits': self.collector.get_retransmits(),
            'timestamp': time.time()
        }
        
        # Validate metrics
        if None in metrics.values():
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
            while True:
                try:
                    # Collect and validate metrics
                    metrics = self._get_metrics()
                    
                    # Make prediction
                    predicted_cca = self.selector.predict(metrics)
                    
                    # Switch CCA if needed
                    if predicted_cca != self.current_cca:
                        set_cca(predicted_cca)
                        self.current_cca = predicted_cca
                        self.collector.log_switch(predicted_cca, metrics)
                    
                    # Periodic retraining
                    if self._should_retrain():
                        print("Retraining model...")
                        self.collector.generate_training_data()
                        CCAModelTrainer().train('data/cca_training_data.parquet')
                        self.selector.reload_model()
                    
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            print("Gracefully shutting down...")
            self.collector.cleanup()
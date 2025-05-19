import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU if not needed
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Optional

class CCASelector:
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.scaler = None
        self.le = LabelEncoder()
        self.model = None
        self.buffer = []  # For LSTM sequence handling
        self.sequence_length = 10  # For LSTM models
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load preprocessing artifacts and model"""
        base_path = "/home/tejasvi/Desktop/CN EL 4th sem/models/"
        try:
            # Load scaler and label encoder
            print(f"Loading scaler from: {base_path}scaler.pkl")  # Debug
            self.scaler = joblib.load(base_path + 'scaler.pkl')

            
            
            print(f"Loading encoder from: {base_path}label_encoder.pkl")  # Debug
            self.le = joblib.load(base_path + 'label_encoder.pkl')
            
            # Load appropriate model
            if self.model_type == 'rf':
                print(f"Loading model from: {base_path}rf_model.pkl")  # Debug
                self.model = joblib.load(base_path + 'rf_model.pkl')
            elif self.model_type == 'lstm':
                self.model = tf.keras.models.load_model('home/tejasvi/Desktop/CN EL 4th sem/models/lstm_model.keras')
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
            
                    
        except FileNotFoundError as e:
            raise RuntimeError("Model artifacts not found. Train models first!") from e

    def _preprocess_features(self, metrics: Dict) -> np.ndarray:
        """Convert raw metrics to ML features"""
        # Base features
        features = [
            metrics['rtt'],
            metrics['throughput'],
            metrics['loss'],
            metrics.get('retransmits', 0)
        ]
        
        # Network stiffness (Throughput × RTT)
        features.append(metrics['throughput'] * metrics['rtt'])
        
        # Stability features (simplified real-time calculation)
        if hasattr(self, 'prev_rtt'):
            rtt_stability = abs(metrics['rtt'] - self.prev_rtt)
            throughput_stability = abs(metrics['throughput'] - self.prev_throughput)
        else:
            rtt_stability = 0.0
            throughput_stability = 0.0
            
        features.extend([rtt_stability, throughput_stability])
        
        # Update previous values for next calculation
        self.prev_rtt = metrics['rtt']
        self.prev_throughput = metrics['throughput']
        
        # Application profile (0=bulk, 1=interactive)
        app_profile = 1 if metrics['throughput'] < 100 else 0
        features.append(app_profile)
        
        return np.array(features).reshape(1, -1)

    def _validate_metrics(self, metrics: Dict) -> bool:
        """Check for valid metric values"""
        required_fields = ['rtt', 'throughput', 'loss']
        return all(
            field in metrics and 
            metrics[field] is not None and 
            metrics[field] >= 0 
            for field in required_fields
        )

    def predict(self, metrics: Dict) -> str:
        """Predict optimal CCA for given metrics"""
        if not self._validate_metrics(metrics):
            raise ValueError("Invalid or missing metric values")
            
        try:
            # Feature preprocessing
            features = self._preprocess_features(metrics)
            scaled_features = self.scaler.transform(features)
            
            if self.model_type == 'rf':
                prediction = self.model.predict(scaled_features)[0]
            elif self.model_type == 'lstm':
                # Maintain sequence buffer
                self.buffer.append(scaled_features[0])
                if len(self.buffer) > self.sequence_length:
                    self.buffer.pop(0)
                
                if len(self.buffer) == self.sequence_length:
                    sequence = np.array(self.buffer).reshape(1, self.sequence_length, -1)
                    prediction = np.argmax(self.model.predict(sequence, verbose=0)[0])
                else:
                    return 'cubic'  # Default until buffer fills
                    
            return self.le.inverse_transform([prediction])[0]
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return 'cubic'  # Fallback to default

    def reload_model(self):
        """Hot-reload model without restarting service"""
        self._load_artifacts()
        if self.model_type == 'lstm':
            self.buffer = []  # Reset sequence buffer

if __name__ == "__main__":
    # Example usage
    selector = CCASelector(model_type='rf')
    
    # Sample metrics (should match your data structure)
    test_metrics = {
        'rtt': 45.6,
        'throughput': 88.2,
        'loss': 1.7,
        'retransmits': 2
    }
    
    predicted_cca = selector.predict(test_metrics)
    print(f"Recommended CCA: {predicted_cca}")
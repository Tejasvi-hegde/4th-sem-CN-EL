import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

class CCADataCollector:
    def __init__(self):
        # Initialize InfluxDB client
        self.client = InfluxDBClient(
            url="http://localhost:8086",
            token="JU8UDuL3viuNAVfv6tRI_HpCXH-oO4RaF5JyOAYv7qzGuOpPNt8EWSy5blc5YuCYwEjxAI50rotWYErz2LzjyA==",
            org="network_monitor"
        )
        self.query_api = self.client.query_api()
        self.bucket = "cca_metrics"
        
        # Configuration
        self.window_size = 5  # Minutes for rolling metrics
        self.history_days = 30  # Data collection period

    def _query_influx(self, query):
        """Execute Flux query and return DataFrame"""
        try:
            result = self.query_api.query_data_frame(query)
            if not result.empty:
                # Convert timestamps and set index
                result['_time'] = pd.to_datetime(result['_time'])
                return result.set_index('_time')
            return pd.DataFrame()
        except Exception as e:
            print(f"Query failed: {e}")
            return pd.DataFrame()

    def get_raw_metrics(self):
        """Retrieve base metrics from InfluxDB"""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{self.history_days}d)
          |> filter(fn: (r) => r._measurement == "network_metrics")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        return self._query_influx(query)

    def calculate_features(self, df):
        """Feature engineering pipeline"""
        if df.empty:
            return df

        # Base features
        features = df[['rtt', 'throughput', 'loss', 'retransmits']].copy()
        
        # Network stiffness (Throughput * RTT)
        features['network_stiffness'] = df['throughput'] * df['rtt']
        
        # Stability metrics (rolling standard deviation)
        rolling_window = f"{self.window_size}T"
        for metric in ['rtt', 'throughput']:
            features[f'{metric}_stability'] = (
                df[metric]
                .rolling(rolling_window, min_periods=10)
                .std()
                .fillna(0)
            )
        
        # Application profile (0=bulk, 1=interactive)
        features['app_profile'] = (
            df['throughput']
            .rolling('1T')
            .std()
            .apply(lambda x: 1 if x > 10 else 0)
        )
        
        # Time-based features
        features['hour_of_day'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        return features.dropna()

    def label_data(self, df):
        """Determine optimal CCA labels (simulated ground truth)"""
        # Temporary rule-based labeling - replace with actual performance analysis
        conditions = [
            (df['rtt'] > 50) | (df['bufferbloat'] > 30),
            df['loss'] > 2,
            df['throughput'] < 100
        ]
        choices = ['bbr', 'westwood', 'cubic']
        
        df['best_cca'] = np.select(conditions, choices, default='cubic')
        return df

    def calculate_bufferbloat(self, df):
        """Calculate bufferbloat from RTT history"""
        if 'rtt' not in df.columns:
            return df
        
        # Calculate 95th percentile RTT - min RTT in 5-minute windows
        df['bufferbloat'] = (
            df['rtt']
            .rolling('5T')
            .apply(lambda x: np.percentile(x, 95) - np.min(x) if len(x) > 10 else 0)
        )
        return df

    def generate_training_data(self, output_path="data/training_data.parquet"):
        """Full data generation pipeline"""
        try:
            print("Collecting raw metrics...")
            raw_df = self.get_raw_metrics()
            
            if raw_df.empty:
                raise ValueError("No data retrieved from InfluxDB")
            
            print("Calculating features...")
            feature_df = self.calculate_features(raw_df)
            
            print("Calculating bufferbloat...")
            bufferbloat_df = self.calculate_bufferbloat(feature_df)
            
            print("Labeling data...")
            labeled_df = self.label_data(bufferbloat_df)
            
            print(f"Saving training data to {output_path}")
            labeled_df.to_parquet(output_path)
            
            return labeled_df
            
        except Exception as e:
            print(f"Data generation failed: {e}")
            return pd.DataFrame()

    def get_live_metrics(self):
        """Get current metrics for real-time prediction"""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -5m)
          |> filter(fn: (r) => r._measurement == "network_metrics")
          |> last()
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df = self._query_influx(query)
        return df.iloc[-1].to_dict() if not df.empty else None

if __name__ == "__main__":
    collector = CCADataCollector()
    
    # Test data collection
    training_data = collector.generate_training_data()
    if not training_data.empty:
        print(f"Generated training data with {len(training_data)} samples")
        print(training_data[['rtt', 'throughput', 'best_cca']].head())
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi
from pathlib import Path

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
            # Handle case where query returns multiple tables
            result = self.query_api.query_data_frame(query)
            
            if isinstance(result, list):
                # Concatenate multiple tables
                df = pd.concat(result, ignore_index=True)
            else:
                df = result

            if not df.empty and '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'])
                df.set_index('_time', inplace=True)
                df = df.sort_index().reset_index().drop_duplicates('_time').set_index('_time')
                return df
            else:
                print("Warning: Query returned empty or invalid data")
                return pd.DataFrame()

        except Exception as e:
            print(f"Query failed: {str(e)}")
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
    
    # In data_collector.py
    def calculate_performance(self):
        """Calculate current network performance score (0-1)"""
        metrics = self.get_live_metrics()
        if not metrics:
            return 0.0
            
        # Simple performance metric (adjust weights as needed)
        score = 0.7 * (metrics['throughput']/1000) +  0.2 * (1 - metrics['loss']/100) +  0.1 * (1 - metrics['rtt']/500)       
                
        return max(0.0, min(1.0, score))  # Clamp to 0-1 range

    def calculate_features(self, df):
        """Feature engineering pipeline"""
        if df.empty:
            return df

        # Sort index first
        df = df.sort_index()

        # Base features
        features = df[['rtt', 'throughput', 'loss', 'retransmits']].copy()
        
        # Network stiffness (Throughput * RTT)
        features['network_stiffness'] = df['throughput'] * df['rtt']
        
        # Stability metrics (rolling standard deviation)
        rolling_window = f"{self.window_size}min"
        for metric in ['rtt', 'throughput']:
            # Handle non-monotonic index
            features[f'{metric}_stability'] = (
                df[metric]
                .sort_index()  # Ensure sorted index
                .rolling(rolling_window, min_periods=10)
                .std()
                .fillna(0)
            )
        
        # Application profile
        features['app_profile'] = (
            df['throughput']
            .sort_index()
            .rolling('1min')
            .std()
            .apply(lambda x: 1 if x > 10 else 0)
        )
        
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
            .rolling('5min')
            .apply(lambda x: np.percentile(x, 95) - np.min(x) if len(x) > 10 else 0)
        )
        return df

    def generate_training_data(self, output_path="data/training_data.parquet"):
        """Full data generation pipeline"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
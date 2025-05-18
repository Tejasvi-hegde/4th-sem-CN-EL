import time
import json
import logging
import sys
sys.path.append("/home/tejasvi/Desktop/CN EL 4th sem/")
from pathlib import Path
from typing import Dict, Optional
from monitor import NetworkMonitor
from ml.inference import CCASelector
import subprocess
class DecisionEngine:
    def __init__(self, use_ml: bool = True):
        self.monitor = NetworkMonitor()
        self.ml_selector = CCASelector(model_type='rf') if use_ml else None
        self.current_cca = self.monitor.get_current_cca()
        self.last_switch_time = time.time()
        self.state_file = Path("decision_state.json")
        self._init_logging()
        self._load_state()

        # Configuration
        self.hysteresis_period = 120  # Seconds between allowed switches
        self.decision_history = []
        self.history_size = 5  # Maintain last 5 decisions

    def _init_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("decisions.log"),
                logging.StreamHandler()
            ]
        )

    def _load_state(self):
        """Load previous state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_cca = state.get('current_cca', 'cubic')
                    self.last_switch_time = state.get('last_switch_time', time.time())
        except Exception as e:
            logging.error(f"Failed to load state: {e}")

    def _save_state(self):
        """Persist current state to file"""
        state = {
            'current_cca': self.current_cca,
            'last_switch_time': self.last_switch_time
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")

    def _rule_based_decision(self, metrics: Dict) -> str:
        """Fallback rule-based decision logic"""
        reasons = []
        
        if metrics.get('bufferbloat', 0) > 50:
            reasons.append("high bufferbloat")
            return 'bbr', reasons
            
        if metrics.get('loss', 0) > 2:
            reasons.append("high packet loss")
            return 'westwood', reasons
            
        if metrics.get('throughput', 0) < 100:
            reasons.append("low throughput")
            return 'cubic', reasons
            
        reasons.append("default policy")
        return 'cubic', reasons

    def _check_hysteresis(self) -> bool:
        """Check if enough time has passed since last switch"""
        elapsed = time.time() - self.last_switch_time
        return elapsed >= self.hysteresis_period

    def _update_history(self, decision: str):
        """Maintain decision history for consistency checks"""
        self.decision_history.append(decision)
        if len(self.decision_history) > self.history_size:
            self.decision_history.pop(0)

    def _consistent_decision(self, new_cca: str) -> bool:
        """Check if decision is consistent with recent history"""
        return all(d == new_cca for d in self.decision_history[-self.history_size:])

    def decide(self) -> Optional[str]:
        """Make a congestion control decision"""
        try:
            metrics = self.monitor.collect_all_metrics()
            logging.info(f"Current metrics: {metrics}")

            new_cca = None
            reason = []
            
            # First try ML-based decision
            if self.ml_selector:
                try:
                    new_cca = self.ml_selector.predict(metrics)
                    reason.append("ML prediction")
                except Exception as e:
                    logging.error(f"ML decision failed: {e}, falling back to rules")

            # Fallback to rule-based decision
            if not new_cca:
                new_cca, reason = self._rule_based_decision(metrics)

            # Check decision consistency and hysteresis
            if (new_cca != self.current_cca and 
                self._check_hysteresis() and 
                self._consistent_decision(new_cca)):
                
                logging.info(f"Switching CCA from {self.current_cca} to {new_cca}"
                            f" - Reason: {', '.join(reason)}")
                self._switch_cca(new_cca)
                return new_cca

            logging.debug("No CCA change needed")
            return None

        except Exception as e:
            logging.error(f"Decision failed: {e}")
            return None

    def _switch_cca(self, new_cca: str):
        """Execute CCA switch and update state"""
        try:
            # Actual switching command
            cmd = (f"ip netns exec {self.monitor.client_ns} "
                  f"sysctl -w net.ipv4.tcp_congestion_control={new_cca}")
            subprocess.run(cmd, shell=True, check=True)
            
            # Update state
            self.current_cca = new_cca
            self.last_switch_time = time.time()
            self._update_history(new_cca)
            self._save_state()
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to switch CCA: {e}")

    def run(self, interval: int = 10):
        """Continuous decision loop"""
        logging.info("Starting decision engine")
        try:
            while True:
                self.decide()
                time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("Stopping decision engine")

if __name__ == "__main__":
    # Example usage
    engine = DecisionEngine(use_ml=True)
    
    # Test decision making
    test_metrics = {
        'rtt': 65.2,
        'throughput': 75.0,
        'loss': 0.5,
        'bufferbloat': 55.0
    }
    print("Test Decision:", engine.decide())
    
    # Run continuous monitoring
    # engine.run(interval=10)
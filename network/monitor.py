import subprocess
import time
import json
import numpy as np
from typing import Dict, Optional
import sys
sys.path.append("/home/tejasvi/Desktop/CN EL 4th sem/")
from utils.influx_client import InfluxDBWriter 
class NetworkMonitor:
    def __init__(self, 
                 client_namespace: str = "ns-client",
                 server_namespace: str = "ns-server",
                 interface: str = "veth-client",
                 target_ip: str = "10.0.0.2"):
        self.client_ns = client_namespace
        self.server_ns = server_namespace
        self.interface = interface
        self.target_ip = target_ip
        self.ping_count = 3  # Default ping attempts
        self.iperf_duration = 2  # Seconds for throughput tests
        self.writer = InfluxDBWriter()

    def _run_command(self, command: str) -> Optional[str]:
        """Execute shell command with error handling"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e.stderr}")
            return None

    def get_rtt(self) -> Optional[float]:
        """Measure average RTT using ping"""
        cmd = (f"ip netns exec {self.client_ns} ping -c {self.ping_count} "
               f"{self.target_ip} | awk -F'/' '/rtt/ {{print $5}}'")
        output = self._run_command(cmd)
        return float(output) if output else None

    def get_throughput(self) -> Optional[float]:
        """Measure network throughput using iperf3 (Mbps)"""
        # Start iperf server in server namespace
        server_cmd = f"ip netns exec {self.server_ns} iperf3 -s -1"
        server = subprocess.Popen(server_cmd, shell=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        time.sleep(1)  # Wait for server to start
        
        # Run iperf client
        client_cmd = (f"ip netns exec {self.client_ns} iperf3 -c {self.target_ip} "
                     f"-t {self.iperf_duration} --json")
        try:
            result = subprocess.run(client_cmd, shell=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return data['end']['sum_received']['bits_per_second'] / 1e6
        except (json.JSONDecodeError, KeyError):
            return None
        finally:
            server.kill()

    def get_loss(self) -> Optional[float]:
        """Get packet loss percentage from tc"""
        try:
            cmd = (f"ip netns exec {self.client_ns} tc -s qdisc show dev {self.interface} "
                "| awk '/loss/ {print $7}' | cut -d'%' -f1")
            output = self._run_command(cmd)
            return float(output) if output else 0.0
        except:
            return 0.0

    def get_bufferbloat(self, test_duration: int = 10) -> Optional[float]:
        """Measure bufferbloat as 95th percentile RTT - min RTT"""
        ping_cmd = (f"ip netns exec {self.client_ns} ping {self.target_ip} "
                   f"-c {test_duration} -i 0.5 | awk -F'=' '/time=/ {{print $2}}' "
                   "| awk '{print $1}'")
        
        # Start background traffic
        iperf_cmd = (f"ip netns exec {self.client_ns} iperf3 -c {self.target_ip} "
                    f"-t {test_duration} --parallel 4")
        iperf_proc = subprocess.Popen(iperf_cmd, shell=True,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        
        # Collect ping data
        try:
            ping_output = self._run_command(ping_cmd)
            if not ping_output:
                return None
                
            rtt_values = [float(t) for t in ping_output.splitlines()]
            return np.percentile(rtt_values, 95) - np.min(rtt_values)
        finally:
            iperf_proc.kill()

    def get_retransmits(self) -> Dict[str, int]:
        """Get TCP retransmission statistics"""
        cmd = f"ip netns exec {self.client_ns} cat /proc/net/netstat"
        output = self._run_command(cmd)
        if not output:
            return {'retransmits': 0, 'fast_retrans': 0}
            
        tcp_ext = {}
        for line in output.split('\n'):
            if line.startswith('TcpExt:'):
                parts = line.strip().split()
                if not tcp_ext:
                    keys = parts[1:]
                else:
                    values = [int(v) for v in parts[1:]]
                    tcp_ext = dict(zip(keys, values))
                    
        return {
            'retransmits': tcp_ext.get('TCPLostRetransmit', 0),
            'fast_retrans': tcp_ext.get('TCPFastRetransmit', 0)
        }

    def get_current_cca(self) -> str:
        """Get currently used congestion control algorithm"""
        cmd = f"ip netns exec {self.client_ns} sysctl -n net.ipv4.tcp_congestion_control"
        output = self._run_command(cmd)
        return output if output else "unknown"

    def collect_all_metrics(self) -> Dict:
        """Collect comprehensive network metrics"""
        metrics= {
            'rtt': self.get_rtt(),
            'throughput': self.get_throughput(),
            'loss': self.get_loss(),
            'bufferbloat': self.get_bufferbloat(),
            'retransmits': self.get_retransmits().get('retransmits', 0),
            'current_cca': self.get_current_cca(),
            'timestamp': time.time()
        }
        self.writer.write(metrics)  # This sends data to InfluxDB
        return metrics

if __name__ == "__main__":
    monitor = NetworkMonitor()
    
    print("Testing Network Monitoring:")
    print(f"Current CCA: {monitor.get_current_cca()}")
    print(f"RTT: {monitor.get_rtt()} ms")
    print(f"Throughput: {monitor.get_throughput()} Mbps")
    print(f"Packet Loss: {monitor.get_loss()}%")
    print(f"Bufferbloat: {monitor.get_bufferbloat()} ms")
    print(f"Retransmits: {monitor.get_retransmits()}")
    
    print("\nComplete Metrics Snapshot:")
    print(monitor.collect_all_metrics())
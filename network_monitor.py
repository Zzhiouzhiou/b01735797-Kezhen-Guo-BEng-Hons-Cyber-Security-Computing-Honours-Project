import pandas as pd
import threading
import time
import socket
import struct
import psutil
import numpy as np
from datetime import datetime


class NetworkMonitor:
    def __init__(self, callback=None, interface=None, window_size=100):
        self.callback = callback
        self.interface = interface
        self.window_size = window_size
        self.running = False
        self.capture_thread = None
        self.available_interfaces = self.get_windows_interfaces()

    def get_windows_interfaces(self):
        interfaces = []

        try:
            # Get all network interfaces using psutil
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()

            for interface_name, interface_addresses in net_if_addrs.items():
                # Skip loopback and disconnected interfaces
                if interface_name in net_if_stats and net_if_stats[interface_name].isup:
                    # Get IPv4 address if available
                    ipv4_address = None
                    for address in interface_addresses:
                        if address.family == socket.AF_INET:
                            ipv4_address = address.address
                            break

                    interfaces.append({
                        'name': interface_name,
                        'ip': ipv4_address or 'Unknown'
                    })

            print(f"Found {len(interfaces)} network interfaces")
        except Exception as e:
            print(f"Error getting network interfaces: {e}")

        return interfaces

    def set_interface(self, interface_name):
        self.interface = interface_name
        print(f"Set interface to: {interface_name}")

    def test_capture(self):
        if not self.interface:
            print("No interface selected")
            return

        print(f"Testing capture on interface: {self.interface}")
        try:
            # Create a raw socket to test if we can capture packets
            # This doesn't actually capture packets, just tests if we can
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            print("Socket created successfully")

            # Find the interface IP
            interface_ip = None
            for iface in self.available_interfaces:
                if iface['name'] == self.interface:
                    interface_ip = iface['ip']
                    break

            if not interface_ip or interface_ip == 'Unknown':
                print("Could not find IP for selected interface")
                return

            # Bind to the interface
            s.bind((interface_ip, 0))
            print(f"Successfully bound to interface {self.interface} with IP {interface_ip}")

            # Close the socket
            s.close()

            print("Interface test successful")
            return True
        except Exception as e:
            print(f"Interface test failed: {e}")
            return False

    def start_capture(self):
        if self.running:
            print("Capture already running")
            return

        if not self.interface:
            raise ValueError("No interface selected")

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("Capture started")

    def stop_capture(self):
        if not self.running:
            print("Capture not running")
            return

        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Capture stopped")

    def _capture_loop(self):
        print("Capture loop started")

        # In a real implementation, this would use packet capture libraries
        # For this demo, we'll generate synthetic data
        while self.running:
            try:
                # Generate synthetic network data
                data = self._generate_test_data()

                # Process the data
                if self.callback and data is not None:
                    self.callback(data)

                # Small delay to prevent high CPU usage
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1)  # Delay before retry

    def _generate_test_data(self):
        """Generate synthetic network data for testing"""
        try:
            # Number of packets to generate
            num_packets = np.random.randint(1, 5)

            # Empty dataframe for storing packet data
            columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                       'num_access_files', 'num_outbound_cmds', 'is_host_login',
                       'is_guest_login', 'count', 'srv_count', 'serror_rate',
                       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                       'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                       'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                       'dst_host_srv_rerror_rate', 'src_ip', 'dst_ip']

            df = pd.DataFrame(columns=columns)

            # Generate random data for each packet
            for i in range(num_packets):
                packet = {}

                # Generate random IPs
                packet['src_ip'] = f"192.168.1.{np.random.randint(1, 255)}"
                packet['dst_ip'] = f"10.0.0.{np.random.randint(1, 255)}"

                # 减少攻击的概率，使大多数流量为正常流量
                attack_probability = 0.05  # 5% chance of attack

                random_value = np.random.random()
                if random_value < attack_probability:
                    # 生成攻击数据包 - 非常明显的差异
                    packet['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'])
                    packet['service'] = np.random.choice(['http', 'ftp', 'smtp', 'ssh'])
                    packet['flag'] = np.random.choice(['REJ', 'RSTO', 'RSTOS0'])

                    # 设置明显可疑的值
                    packet['count'] = np.random.randint(300, 500)
                    packet['serror_rate'] = np.random.uniform(0.85, 1.0)
                    packet['same_srv_rate'] = np.random.uniform(0.0, 0.2)
                    packet['wrong_fragment'] = np.random.randint(1, 3)
                    packet['urgent'] = np.random.randint(1, 3)
                    packet['num_failed_logins'] = np.random.randint(3, 5)
                else:
                    # 生成正常流量 - 更明确地标记为正常
                    packet['protocol_type'] = np.random.choice(['tcp', 'udp'], p=[0.8, 0.2])
                    packet['service'] = np.random.choice(['http', 'https', 'dns'], p=[0.4, 0.4, 0.2])
                    packet['flag'] = 'SF'  # 正常连接

                    # 设置正常值
                    packet['count'] = np.random.randint(1, 5)
                    packet['serror_rate'] = 0.0
                    packet['same_srv_rate'] = np.random.uniform(0.8, 1.0)
                    packet['wrong_fragment'] = 0
                    packet['urgent'] = 0
                    packet['num_failed_logins'] = 0

                # 通用字段
                packet['duration'] = np.random.randint(1, 1000)
                packet['src_bytes'] = np.random.randint(100, 10000)
                packet['dst_bytes'] = np.random.randint(100, 10000)
                packet['land'] = 0
                packet['logged_in'] = 1

                # 生成其余字段的值
                for col in columns:
                    if col not in packet:
                        if col.endswith('rate'):
                            packet[col] = np.random.uniform(0, 1)
                        else:
                            packet[col] = np.random.randint(0, 100)

                # 添加到dataframe
                df.loc[i] = packet

            return df

        except Exception as e:
            print(f"Error generating test data: {e}")
            return None
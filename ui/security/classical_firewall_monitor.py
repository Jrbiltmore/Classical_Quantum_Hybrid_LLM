
import socket

def monitor_firewall():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(('8.8.8.8', 53))
        return 'Connected to firewall'
    except socket.error as e:
        return f'Firewall monitoring failed: {e}'

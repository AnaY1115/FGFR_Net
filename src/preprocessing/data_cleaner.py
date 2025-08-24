'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import re
import logging
import numpy as np
from scapy.all import rdpcap, Packet
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, max_payload_length=1500, remove_empty=True):
        self.max_payload_length = max_payload_length
        self.remove_empty = remove_empty

        self.protocol_ports = {
            'http': [80, 8080],
            'https': [443, 8443],
            'ftp': [20, 21],
            'ssh': [22],
            'dns': [53],
            'smtp': [25, 587],
            'pop3': [110],
            'imap': [143, 993]
        }
    
    def _extract_payload(self, packet):

        payload = b""
        
        if hasattr(packet, 'payload'):
            current = packet.payload
            while hasattr(current, 'payload') and current.payload:
                if current.name in ['TCP', 'UDP']:
                    payload = bytes(current.payload)
                    break
                current = current.payload
        
        return payload
    
    def _truncate_payload(self, payload):

        if len(payload) > self.max_payload_length:
            return payload[:self.max_payload_length]
        return payload
    
    def _normalize_payload(self, payload):

        if not payload:
            return np.array([])
            
        payload_array = np.frombuffer(payload, dtype=np.uint8)
        return payload_array / 255.0
    
    def _get_protocol_from_port(self, packet):

        if 'TCP' in packet:
            src_port = packet['TCP'].sport
            dst_port = packet['TCP'].dport
        elif 'UDP' in packet:
            src_port = packet['UDP'].sport
            dst_port = packet['UDP'].dport
        else:
            return None
            
        for proto, ports in self.protocol_ports.items():
            if src_port in ports or dst_port in ports:
                return proto
        return None
    
    def clean_single_pcap(self, input_pcap, output_file=None):
        try:
            packets = rdpcap(input_pcap)
            
            if self.remove_empty and len(packets) == 0:
                logger.debug(f"空pcap文件: {input_pcap}")
                return None
            
            session_data = {
                'filename': os.path.basename(input_pcap),
                'packet_count': len(packets),
                'protocols': set(),
                'payloads': [],
                'normalized_payloads': [],
                'total_length': 0
            }
            
            for packet in packets:
                proto = self._get_protocol_from_port(packet)
                if proto:
                    session_data['protocols'].add(proto)
                
                payload = self._extract_payload(packet)
                if payload:
                    truncated_payload = self._truncate_payload(payload)
                    normalized_payload = self._normalize_payload(truncated_payload)
                    
                    session_data['payloads'].append(truncated_payload)
                    session_data['normalized_payloads'].append(normalized_payload)
                    session_data['total_length'] += len(truncated_payload)
            
            session_data['protocols'] = list(session_data['protocols'])
            
            if output_file:
                Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
                np.savez(output_file, **session_data)
                logger.debug(f"清洗后的文件已保存: {output_file}")
            
            return session_data
            
        except Exception as e:
            logger.error(f"处理pcap文件{input_pcap}时出错: {str(e)}")
            return None
    
    def batch_clean(self, input_dir, output_dir, recursive=True):
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(input_dir):
            pcap_files = [f for f in files if f.endswith(('.pcap', '.pcapng'))]
            total_count += len(pcap_files)
            
            if pcap_files:
                relative_path = os.path.relpath(root, input_dir)
                current_output_dir = os.path.join(output_dir, relative_path)
                
                for pcap_file in pcap_files:
                    input_path = os.path.join(root, pcap_file)
                    output_filename = os.path.splitext(pcap_file)[0] + '.npz'
                    output_path = os.path.join(current_output_dir, output_filename)
                    
                    result = self.clean_single_pcap(input_path, output_path)
                    if result:
                        success_count += 1
        
        logger.info(f"批量清洗完成，共处理{total_count}个文件，成功{success_count}个，失败{total_count - success_count}个")
        return success_count

if __name__ == "__main__":
    cleaner = DataCleaner(max_payload_length=1500)
    
    # cleaner.clean_single_pcap("split_sessions/session1.pcap", "cleaned_data/session1.npz")
    
    # cleaner.batch_clean("split_data/iscx", "cleaned_data/iscx")

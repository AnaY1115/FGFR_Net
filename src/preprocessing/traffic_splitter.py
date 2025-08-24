'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSplitter:
    def __init__(self, splitcap_path=None):
s
        self.splitcap_path = splitcap_path if splitcap_path else "SplitCap"
        self._verify_splitcap()
    
    def _verify_splitcap(self):
        try:
            if os.name == 'nt':  # Windows系统
                result = subprocess.run([self.splitcap_path, "-h"], check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       shell=True)
            else:  # Linux/Mac系统
                result = subprocess.run([self.splitcap_path, "-h"], check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("SplitCap验证成功")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("SplitCap验证失败，请确保SplitCap已正确安装并在环境变量中，或指定正确路径")
            raise
    
    def split_pcap(self, input_pcap, output_dir, split_by="session"):
        if not os.path.exists(input_pcap):
            logger.error(f"输入pcap文件不存在: {input_pcap}")
            raise FileNotFoundError(f"输入pcap文件不存在: {input_pcap}")
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"开始分割pcap文件: {input_pcap}")
            
            command = [
                self.splitcap_path,
                "-r", input_pcap,
                "-o", output_dir,
                "-s", split_by
            ]
            
            if os.name == 'nt':
                result = subprocess.run(" ".join(command), check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       shell=True, text=True)
            else:
                result = subprocess.run(command, check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
            
            output_files = [f for f in os.listdir(output_dir) if f.endswith('.pcap')]
            logger.info(f"pcap文件分割完成，生成{len(output_files)}个会话文件，保存至: {output_dir}")
            
            return len(output_files)
            
        except subprocess.SubprocessError as e:
            logger.error(f"分割pcap文件失败: {str(e)}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def batch_split(self, input_dir, output_root, recursive=False):

        total_files = 0
        
        for root, dirs, files in os.walk(input_dir):
            pcap_files = [f for f in files if f.endswith(('.pcap', '.pcapng'))]
            
            if pcap_files:
                relative_path = os.path.relpath(root, input_dir)
                output_dir = os.path.join(output_root, relative_path)
                
                for pcap_file in pcap_files:
                    input_pcap = os.path.join(root, pcap_file)
                    file_output_dir = os.path.join(output_dir, os.path.splitext(pcap_file)[0])
                    count = self.split_pcap(input_pcap, file_output_dir)
                    total_files += count
            
            if not recursive:
                break
        
        logger.info(f"批量分割完成，共生成{total_files}个会话文件")
        return total_files

if __name__ == "__main__":
    splitter = TrafficSplitter()
    
    # splitter.split_pcap("input.pcap", "output_sessions")
    
    # splitter.batch_split("raw_data/iscx", "split_data/iscx", recursive=True)

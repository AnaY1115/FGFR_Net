# FGFR-Net: An advanced traffic classification method

With the rapid development of encryption technologies (such as TLS 1.3, SSL 3.0) and the exponential growth of network applications, accurate Encrypted Traffic Classification (ETC) is crucial for network security, intrusion detection, Quality of Service/Quality of Experience (QoS/QoE) management, and network visibility. This project implements a novel encrypted traffic classification model based on byte-level traffic graphs and improved residual networks, achieving state-of-the-art performance on the public datasets ISCX-VPNonVPN and USTC-TFC.

## Model Architecture
1. Data preprocessing module: traffic normalization, truncation, cleaning, segmentation, etc.
2. Byte-level traffic graph construction module: constructing traffic into graphs
3. Feature encoding module: capturing and building traffic features
4. Classifier: multi-class classification

## Installation and Usage

### Environment Requirements
- Python 3.8+
- PyTorch 1.10.1
- tshark
- SplitCap
- Scapy
- Dependent libraries: numpy, pandas, scikit-learn, torch-geometric

### Installation Steps
1. Clone the repository:

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure dataset paths in `config/dataset_config.json`:
    - Download the ISCX-VPNonVPN and USTC-TFC datasets
    - Set `iscx_path` and `ustc_path` to local dataset directories

4. Configure model parameters (batch size, epochs, learning rate, etc.) in `config/model_config.json`

### Usage Methods
1. Preprocess the dataset:
```bash
python src/preprocessing/run_preprocess.py --dataset iscx  # or ustc
```

2. Train the model:
```bash
python src/trainer.py --dataset iscx --epochs 120 --batch_size 256
```

## Directory Structure
```
FGFR-Net/
├── requirements.txt         
├── README.md                
├── config/                  
│   ├── dataset_config.json 
│   ├── model_config.json     
│   └── train_config.json
├── src/
│   ├── preprocessing/       
│   │   ├── traffic_splitter.py  
│   │   ├── data_cleaner.py     
│   │   └── run_preprocess.py   
│   ├── graph_construction/ 
│   │   ├── node_constructor.py  
│   │   ├── edge_constructor.py  
│   │   └── graph_builder.py   
│   ├── feature_encoding/   
│   │   └── gin_encoder.py   
│   ├── classifier/        
│   │   ├── resnet34.py        
│   │   ├── deformable_conv.py 
│   │   ├── depthwise_conv.py   
│   │   ├── channel_attention.py 
│   │   └── cb_loss.py        
│   ├── trainer.py           
│   └── evaluator.py        
├── models/                 
├── datasets/              
│   ├── iscx/
│   └── ustc/
├── results/                  
│   ├── metrics/
│   ├── confusion_matrices/
│   └── efficiency_analysis/
└── tests/              
    ├── test_graph_construction.py
    ├── test_encoder.py
    └── test_classifier.py
```

## Contribution Guidelines
Contributions are welcome! Please follow these steps:

1. Submit an Issue describing the feature you propose or the bug fix
2. Create a feature branch
3. Commit changes with clear descriptions
4. Submit a Pull Request, ensuring all tests pass

## Notes
- Datasets used: ISCX-VPNonVPN and USTC-TFC (citation required for research use)
- Note that FGFR-Net is currently for research purposes only. Users shall bear full responsibility for any unexpected issues arising during application exploration.
- If you reference this project in research, papers, or other projects, please provide appropriate citations.
- Thank you for respecting and supporting the intellectual property rights of this project. Some core files of the project have been uploaded, and we will continue to organize and update the complete code and documentation.

The code will be continuously updated, so stay tuned!

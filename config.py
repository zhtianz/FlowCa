import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.base_channels = 64
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 8
        self.use_learned_initial = False
        self.dropout_rate = 0.2     
        self.use_batchnorm = True       
        self.feature_dropout = 0.1,       
        
        self.gradient_accumulation_steps = 4  
        self.warmup_epochs = 10
        self.max_grad_norm = 1.0
        
        self.sampling_steps = 50
        
        self.grid_size = 64
        
        self.save_interval = 10
        self.visualize_interval = 10

class TrainConfig(Config):
    """训练专用配置"""
    def __init__(self):
        super().__init__()
        self.batch_size = 16  
        self.num_epochs = 150
        self.learning_rate = 0.00012784342713816085  
        self.weight_decay = 5.348113163629746e-05
        self.dropout_rate = 0.29240121595627766
        self.feature_dropout = 0.16553999496087562 
        self.use_attention = True
        self.base_channels = 64
        self.use_learned_initial = False

        self.info_txt = "train_datasetsid.txt"
        self.data_path = "train_datasets"
        self.data_list = "data_lists_model_soft.pkl"
        self.train_data = "train_data_model_soft.pkl"
        self.valid_data = "valid_data_model_soft.pkl"
        self.log_dir = os.path.join(
            "logs",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.resume_from = ''
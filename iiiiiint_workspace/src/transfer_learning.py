from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from .models import BasePredictor

class TransferEncoder(nn.Module):
    """迁移学习编码器 - 在相似股票数据上预训练"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.projection = nn.Linear(hidden_size // 2, hidden_size // 4)
        
    def forward(self, x):
        features = self.encoder(x)
        projected = self.projection(features)
        return features, projected

class TransferLSTMPredictor(BasePredictor):
    """迁移学习LSTM预测器"""
    
    def __init__(self, name: str, input_shape: Tuple[int, int], config: Dict):
        super().__init__(name)
        seq_len, feat_dim = input_shape
        
        # 编码器部分
        self.encoder = TransferEncoder(feat_dim, 
                                     hidden_size=config.get("transfer_hidden_size", 64),
                                     dropout=config.get("transfer_dropout", 0.2))
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=config.get("transfer_hidden_size", 64) // 2,  # 编码器输出维度
            hidden_size=config.get("lstm_hidden_size", 32),
            num_layers=2,
            batch_first=True,
            dropout=config.get("lstm_dropout", 0.1),
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.get("lstm_hidden_size", 32) * 2, 32),  # 双向LSTM
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.lstm.to(self.device)
        self.classifier.to(self.device)
        
        self.epochs = config.get("transfer_epochs", 10)
        self.lr = config.get("transfer_lr", 1e-4)
        self.batch_size = config.get("transfer_batch_size", 32)
        
        # 预训练状态
        self.is_pretrained = False
        
    def pretrain_encoder(self, source_data: np.ndarray, val_data: np.ndarray = None):
        """在源数据上预训练编码器"""
        if source_data is None or len(source_data) == 0:
            return
            
        print("开始在相似股票数据上预训练编码器...")
        
        # 转换为Tensor
        source_tensor = torch.tensor(source_data, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(source_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 对比学习损失
        criterion = nn.CosineEmbeddingLoss()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        
        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                
                optimizer.zero_grad()
                
                # 正样本对：同一批次中的增强版本
                features1, proj1 = self.encoder(x)
                features2, proj2 = self.encoder(x)  # 可以添加数据增强
                
                # 对比损失
                target = torch.ones(x.size(0)).to(self.device)
                loss = criterion(proj1, proj2, target)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"预训练 Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        self.is_pretrained = True
        print("编码器预训练完成")
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, source_data=None):
        """训练迁移学习模型"""
        if not self.is_pretrained and source_data is not None:
            self.pretrain_encoder(source_data)
        
        if X_train is None or len(X_train) == 0:
            return
            
        # 准备数据
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 优化器
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.lstm.parameters()) + 
            list(self.classifier.parameters()),
            lr=self.lr
        )
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(self.epochs):
            # 训练
            self.encoder.train()
            self.lstm.train()
            self.classifier.train()
            
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                batch_size, seq_len, feat_dim = batch_X.shape
                x_reshaped = batch_X.reshape(-1, feat_dim)
                
                # 编码特征
                with torch.no_grad() if self.is_pretrained else torch.enable_grad():
                    encoded_features, _ = self.encoder(x_reshaped)
                
                encoded_features = encoded_features.reshape(batch_size, seq_len, -1)
                
                # LSTM处理
                lstm_out, _ = self.lstm(encoded_features)
                lstm_last = lstm_out[:, -1, :]
                
                # 分类
                probs = self.classifier(lstm_last)
                loss = criterion(probs, batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            val_loss = None
            if X_valid is not None and len(X_valid) > 0:
                val_loss = self._validate(X_valid, y_valid, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        'encoder': self.encoder.state_dict(),
                        'lstm': self.lstm.state_dict(),
                        'classifier': self.classifier.state_dict()
                    }
            
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss_str}")
        
        if best_state:
            self.encoder.load_state_dict(best_state['encoder'])
            self.lstm.load_state_dict(best_state['lstm'])
            self.classifier.load_state_dict(best_state['classifier'])
    
    def _validate(self, X_valid, y_valid, criterion):
        """验证步骤"""
        self.encoder.eval()
        self.lstm.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            Xv = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            yv = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1).to(self.device)
            
            batch_size, seq_len, feat_dim = Xv.shape
            x_reshaped = Xv.reshape(-1, feat_dim)
            
            encoded_features, _ = self.encoder(x_reshaped)
            encoded_features = encoded_features.reshape(batch_size, seq_len, -1)
            
            lstm_out, _ = self.lstm(encoded_features)
            lstm_last = lstm_out[:, -1, :]
            
            probs = self.classifier(lstm_last)
            loss = criterion(probs, yv).item()
            
        return loss
    
    def predict_proba(self, X) -> np.ndarray:
        """预测概率"""
        if X is None or len(X) == 0:
            return np.array([])
            
        self.encoder.eval()
        self.lstm.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            batch_size, seq_len, feat_dim = tensor.shape
            
            x_reshaped = tensor.reshape(-1, feat_dim)
            encoded_features, _ = self.encoder(x_reshaped)
            encoded_features = encoded_features.reshape(batch_size, seq_len, -1)
            
            lstm_out, _ = self.lstm(encoded_features)
            lstm_last = lstm_out[:, -1, :]
            
            probs = self.classifier(lstm_last).cpu().numpy().reshape(-1)
            
        return probs

def prepare_transfer_data(dm_source, dm_target, config):
    """准备迁移学习数据"""
    # 获取源数据特征
    source_df = dm_source.get_target_df()
    source_features = source_df[config["feature_cols_deep_base"]].values
    
    # 获取目标数据序列
    target_df = dm_target.get_target_df()
    X_target, y_target, _ = dm_target.build_sequence_data(
        target_df, config["feature_cols_deep_base"], "label", config["seq_len"]
    )
    
    return source_features, X_target, y_target
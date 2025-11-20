import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Attention Block (Simple Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels[-1], num_heads=4, batch_first=True)
        
        self.linear = nn.Linear(num_channels[-1], 1)
        # Sigmoid for probability [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Features, Time)
        y = self.network(x) # (Batch, Channels, Time)
        
        # Apply Attention
        # Permute to (Batch, Time, Channels) for MultiheadAttention
        y_perm = y.permute(0, 2, 1)
        attn_out, _ = self.attention(y_perm, y_perm, y_perm)
        
        # Global Average Pooling on Attention Output
        # Permute back to (Batch, Channels, Time)
        y_attn = attn_out.permute(0, 2, 1)
        y_gap = self.gap(y_attn).squeeze(-1)
        
        logits = self.linear(y_gap)
        return self.activation(logits)

class HybridLoss(nn.Module):
    def __init__(self, target_sigma=0.15, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.target_sigma = target_sigma
        self.alpha = alpha # Weight for Sharpe Loss (1-alpha for BCE)
        self.bce = nn.BCELoss()
        
    def forward(self, prob_out, targets, returns, volatility):
        # prob_out: (Batch, 1) -> Probability [0, 1]
        # targets: (Batch, 1) -> Binary Target {0, 1}
        # returns: (Batch, 1) -> Next day return
        # volatility: (Batch, 1) -> Current volatility
        
        # 1. BCE Loss
        bce_loss = self.bce(prob_out, targets)
        
        # 2. Sharpe Loss
        # Convert prob to weight: w = (p - 0.5) * 2 -> [-1, 1]
        raw_signal = (prob_out - 0.5) * 2
        
        # Volatility Scaling
        vol_scale = torch.clamp(self.target_sigma / (volatility + 1e-8), 0, 5.0)
        weights = raw_signal * vol_scale
        
        strategy_returns = weights * returns
        
        mean_ret = torch.mean(strategy_returns)
        std_ret = torch.std(strategy_returns) + 1e-8
        
        sharpe = mean_ret / std_ret
        
        # Combine: Minimize BCE and Maximize Sharpe
        # Loss = (1-alpha)*BCE - alpha*Sharpe
        # Scale Sharpe to be comparable? Sharpe is usually < 1 (daily). BCE is ~0.7.
        # Let's weight them equally roughly.
        
        total_loss = (1 - self.alpha) * bce_loss - self.alpha * sharpe
        
        return total_loss, bce_loss.item(), sharpe.item()

import copy

def train_model(model, train_loader, val_loader, epochs=10, lr=0.0001, device='cpu', alpha=0.5):
    criterion = HybridLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.to(device)
    
    best_sharpe = -float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_sharpe = 0.0
        
        for X_batch, y_batch, ret_batch, vol_batch in train_loader:
            X_batch = X_batch.to(device).permute(0, 2, 1)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            ret_batch = ret_batch.to(device).float().unsqueeze(1)
            vol_batch = vol_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss, bce, sharpe = criterion(outputs, y_batch, ret_batch, vol_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bce += bce
            train_sharpe += sharpe
            
        # Validation
        model.eval()
        val_sharpe = 0.0
        all_rets = []
        with torch.no_grad():
            for X_batch, _, ret_batch, vol_batch in val_loader:
                X_batch = X_batch.to(device).permute(0, 2, 1)
                ret_batch = ret_batch.to(device).float().unsqueeze(1)
                vol_batch = vol_batch.to(device).float().unsqueeze(1)
                
                outputs = model(X_batch)
                
                # Re-calculate strategy returns for validation
                raw_signal = (outputs - 0.5) * 2
                vol_scale = torch.clamp(0.15 / (vol_batch + 1e-8), 0, 5.0)
                weights = raw_signal * vol_scale
                strat_ret = weights * ret_batch
                all_rets.append(strat_ret)
                
        # Compute Val Sharpe over entire validation set
        all_rets = torch.cat(all_rets)
        val_mean = torch.mean(all_rets)
        val_std = torch.std(all_rets) + 1e-8
        val_sharpe = (val_mean / val_std).item() * np.sqrt(252) # Annualized
        
        avg_bce = train_bce / len(train_loader)
        avg_sharpe = train_sharpe / len(train_loader)
        
        # Compute Val Accuracy
        correct = 0
        total = 0
        with torch.no_grad():
             for X_batch, y_batch, _, _ in val_loader:
                X_batch = X_batch.to(device).permute(0, 2, 1)
                y_batch = y_batch.to(device).float().unsqueeze(1)
                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, BCE: {avg_bce:.4f}, Val Sharpe: {val_sharpe:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early Stopping (Save Best Model)
        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_model_state = copy.deepcopy(model.state_dict())
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val Sharpe: {best_sharpe:.4f}")


if __name__ == "__main__":
    # Simple test
    model = TCN(num_inputs=12, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)
    x = torch.randn(32, 12, 64) # (Batch, Features, Time)
    y = model(x)
    print(f"Output shape: {y.shape}")

import torch
from torch import nn
from .net_module import NetVLAGD
from .net_module import NeXtVLAD
from .trans import *

class NonlinearClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        super(NonlinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.project = nn.Linear(feature_dim, num_class)
        self.drop_layer = nn.Dropout(p=0.5)
        self.early = False

    def forward(self, x):
        logits = self.logits(x)
        prediction = torch.sigmoid(logits)
        return prediction, logits

    def logits(self, x):
        if self.early:
            x = torch.mean(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        logits = self.project(x)
        
        if not self.early:
            logits = torch.mean(logits, 1)
        return logits   


class NetVLADModel(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(NetVLADModel, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        cluster_size = 64
        hidden_size = 1024
        self.add_batch_norm = True
        self.relu = True
        
        self.vlad = NetVLAGD(feature_size=self.feature_size,
                                cluster_size=cluster_size,
                                add_batch_norm=True,
                                add_bias=False)

        # classifier
        self.hidden_fc = nn.Linear(cluster_size*self.feature_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.relu1 = nn.ReLU6()

        # classifier
        self.pred_fc = nn.Linear(hidden_size, num_class)
    
    def forward(self, x):
        logits = self.logits(x)
        prediction = torch.sigmoid(logits)
        return prediction, logits
    
    def logits(self, x):
        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)

        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)
            
        vlad = self.vlad(x)
        # activation -> [bs, hidden_size]
        activation = self.hidden_fc(vlad)
        if self.add_batch_norm and self.relu:
            activation = self.bn2(activation)
        
        if self.relu:
            activation = self.relu1(activation)

        logits = self.pred_fc(activation)
        return logits


class NeXtVLADModel(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(NeXtVLADModel, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        cluster_size = 64
        hidden_size = 1024
        self.add_batch_norm = True
        self.relu = True
        
        self.vlad = NeXtVLAD(dim=self.feature_size, num_clusters=cluster_size
        , max_frames=200)

        # classifier
        self.hidden_fc = nn.Linear(int(cluster_size*self.feature_size*2/8), hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.relu1 = nn.ReLU6()

        # classifier
        self.pred_fc = nn.Linear(hidden_size, num_class)
    
    def forward(self, x, y=None, phase='val'):
        logits = self.logits(x)
        prediction = torch.sigmoid(logits)
        return prediction, logits
    
    def logits(self, x):
        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)

        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)
            
        vlad = self.vlad(x)
        # print()
        # activation -> [bs, hidden_size]
        activation = self.hidden_fc(vlad)
        if self.add_batch_norm and self.relu:
            activation = self.bn2(activation)
        
        if self.relu:
            activation = self.relu1(activation)

        logits = self.pred_fc(activation)
        return logits


class Trans_Pos_diff(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(Trans_Pos_diff, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        self.add_batch_norm = True

        self.bn1 = nn.BatchNorm1d(self.feature_size)

        for p in self.parameters():
            p.requires_grad = False

        self.self_attn = trans_zijida_pos()

    
    def forward(self, x, x_diff, y=None, phase='val'):
        logits, z, logits1 = self.logits(x, x_diff, y, phase)
        prediction = torch.sigmoid(logits)
        prediction1 = torch.sigmoid(logits1)

        return prediction, logits, z, prediction1, logits1
    
    def logits(self, x, x_diff, y=None, phase='val'):
        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)

        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)
            
        logits, z, branch = self.self_attn(x, x_diff, y, phase)

        return logits, z, branch


class Trans_Pos_101(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(Trans_Pos_101, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        self.add_batch_norm = True

        self.bn1 = nn.BatchNorm1d(self.feature_size)

        self.self_attn = trans_zijida_pos_101()

    def forward(self, x, y=None, phase='val'):
        logits, z, logits1 = self.logits(x, y, phase)
        prediction = torch.sigmoid(logits)
        prediction1 = torch.sigmoid(logits1)

        return prediction, logits, z, prediction1, logits1
    
    def logits(self, x, y=None, phase='val'):
        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)

        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)
            
        logits, z, branch = self.self_attn(x, y, phase)

        return logits, z, branch

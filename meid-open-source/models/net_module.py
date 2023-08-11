import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAGD(nn.Module):
    def __init__(self, feature_size, cluster_size, add_batch_norm, add_bias=False, gating=True):
        super(NetVLAGD, self).__init__() 
        self.feature_size = feature_size
        self.add_batch_norm = add_batch_norm 
        self.cluster_size = cluster_size 
        
        self.add_bias = add_bias 
        self.gating = gating

        self.activation_fc = nn.Linear(feature_size, cluster_size, bias=add_bias)
        self.gate_weight = nn.Parameter(torch.zeros(cluster_size, feature_size))
        self.gate_weight = nn.init.xavier_normal_(self.gate_weight)
        
        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(cluster_size)
        
    def forward(self, x):
        # x shape: [batch_size*max_frames, feature_size]
        batch_size, max_frames, feature_size = x.shape
        
        x = x.view(batch_size* max_frames, feature_size)
        
        activation = self.activation_fc(x)
        
        if self.add_batch_norm:
            activation = self.bn1(activation)
        
        # [batchsize*max_frames, cluster_size]
        activation = F.softmax(activation, 1)
        activation = activation.view(-1, max_frames, self.cluster_size)
        
        activation = activation.permute(0, 2, 1)
        
        # activation : [batch_size, cluster_size, max_frames]
        # reshaped_input : [batch_size, max_frames, feature_size]
        reshaped_input = x.view(-1, max_frames, feature_size)
        
        # vlagd -> [bs, cluster_size ,feature_size]
        vlagd = torch.matmul(activation, reshaped_input)

        if self.gating:
            gate_weight = F.sigmoid(self.gate_weight)
            vlagd = torch.mul(vlagd, gate_weight)
            # vlagd -> [bs, feature_size, cluster_size]
            vlagd = vlagd.permute(0, 2, 1)
        
        vlagd = F.normalize(vlagd, p=2, dim=1)
        
        # permute breaks the contiguous of tensor, must use reshape
        vlagd = vlagd.reshape(batch_size, self.cluster_size * feature_size)
        vlagd = F.normalize(vlagd, p=2, dim=1)
        return vlagd


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, dim=1024, num_clusters=64, lamb=2, groups=8, max_frames=150):
        super(NeXtVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)
        # expansion FC
        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        #         print(f"x: {x.shape}")

        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)

        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        alpha_g = torch.sigmoid(self.fc_g(x_dot))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)
        return vlad

if __name__ == '__main__':

    aaa = NeXtVLAD(dim=2048)
    input = torch.randn(7,150,2048)

    output = aaa.forward(input)
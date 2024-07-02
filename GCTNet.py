import torch
import torch.nn as nn
from loss import batch_episym
from transformer import TransformerLayer

class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
        
class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class Interation(nn.Module):
    def __init__(self, in_channel):
        nn.Module.__init__(self)

        self.attq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, n1, n2, n3):
        q1 = self.attq(n1).squeeze(3)
        k1 = self.attk(n2).squeeze(3)
        v1 = self.attv(n3).squeeze(3)
        scores = torch.bmm(q1.transpose(1, 2), k1)
        att = torch.softmax(scores, dim=2)
        out = torch.bmm(v1, att.transpose(1, 2))
        out = out.unsqueeze(3)
        out = self.conv(out)
        out = n3 + self.gamma * out
        return out

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    """

    @param x_in: points
    @param logits: weights and mask
    @return: essential matrix
    """
    if logits.shape[1] == 2:
        mask = logits[:, 0, :, 0]
        weights = logits[:, 1, :, 0]

        mask = torch.sigmoid(mask)
        weights = torch.exp(weights) * mask
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    elif logits.shape[1] == 1:
        weights = torch.relu(torch.tanh(logits))  # tanh and relu

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    """Dynamic Grouping

    @param x: correspondence's feature, [feature_dim, N, 1]
    @param k: number of local neighborhood edges
    @param idx:
    @return: dynamic graph, [feature_dim*2, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature



class GCET(nn.Module): #Graph Context Enhance Transformer
    def __init__(self, knn_num=9, in_channel=128,clusters=256):
        super(GCET, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.mlp1=ResNet_Block(in_channel*2, 2*in_channel, pre=True)
        self.change1=ResNet_Block(in_channel*2, in_channel, pre=True)
        self.change2=ResNet_Block(in_channel*2, in_channel, pre=True)
        self.inter1=Interation(in_channel)
        self.inter2=Interation(in_channel)
        self.intra1=Interation(in_channel)
        self.intra2=Interation(in_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel*2, self.in_channel*2, (1, 3), stride=(1, 3)),
            nn.BatchNorm2d(self.in_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel*2, self.in_channel*2, (1, 3)),
            nn.BatchNorm2d(self.in_channel*2),
            nn.ReLU(inplace=True),
        )
        self.conv_group1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, clusters, kernel_size=1)
        )
        self.conv_group2 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, clusters, kernel_size=1)
        )
        self.conv_group3 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, clusters, kernel_size=1)
        )
        self.conv_group4 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, clusters, kernel_size=1)
        )
        self.aff1=AFF(in_channel,r=4)
        self.aff2=AFF(in_channel,r=4)
        self.aff3=AFF(in_channel,r=4)
    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out_an = self.conv(out)
        out_an = self.change1(out_an)
        out_max=self.mlp1(out)
        out_max = out_max.max(dim=-1,keepdim=False)[0]
        out_max = out_max.unsqueeze(3)
        out_max = self.change2(out_max)
        embed1 = self.conv_group1(out_an)
        S1 = torch.softmax(embed1, dim=2).squeeze(3)
        cluster_an = torch.matmul(out_an.squeeze(3), S1.transpose(1, 2)).unsqueeze(3)
        embed2 = self.conv_group2(out_max)
        S2 = torch.softmax(embed2, dim=2).squeeze(3)
        cluster_max = torch.matmul(out_max.squeeze(3), S2.transpose(1, 2)).unsqueeze(3)
        an_inter=self.inter1(cluster_an,cluster_an,cluster_an)
        max_inter=self.inter2(cluster_max,cluster_max,cluster_max)
        an_intra=self.intra1(cluster_an,cluster_max,cluster_max)
        max_intra=self.intra2(cluster_max,cluster_an,cluster_an)
        fusion_an=self.aff1(an_inter,an_intra)
        fusion_max=self.aff2(max_inter,max_intra)
        embed3 = self.conv_group3(out_max)
        S3 = torch.softmax(embed3, dim=1).squeeze(3)
        out_max = torch.matmul(fusion_an.squeeze(3), S3).unsqueeze(3)
        embed4 = self.conv_group4(out_an)
        S4 = torch.softmax(embed4, dim=1).squeeze(3)
        out_an = torch.matmul(fusion_max.squeeze(3), S4).unsqueeze(3)
        out=self.aff3(out_max,out_an)
        return out

class GCGT(nn.Module): #Graph Context Guidance Transformer
    def __init__(self, in_channels=128, clusters=256):
        super(GCGT, self).__init__()

        num_heads = 4
        dropout = None
        activation_fn = 'ReLU'
        self.transformer = TransformerLayer(in_channels, num_heads, dropout=dropout, activation_fn=activation_fn)
        
        self.conv_group1 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, clusters, kernel_size=1)
        )
        self.conv_group2 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, clusters, kernel_size=1)
        )
        self.conv_group3 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, clusters, kernel_size=1)
        )
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(in_channels, eps=1e-3),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(clusters),
                nn.ReLU(),
                nn.Conv2d(clusters, clusters, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(in_channels, eps=1e-3),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1)
                )
        self.aff1=AFF(in_channels,r=4)
        self.aff2=AFF(in_channels,r=4)
        
        
    def forward(self, q, v):
        embed1 = self.conv_group1(q)
        S1 = torch.softmax(embed1, dim=2).squeeze(3)
        clustered_q = torch.matmul(q.squeeze(3), S1.transpose(1, 2)).unsqueeze(3)
        embed2 = self.conv_group2(v)
        S2 = torch.softmax(embed2, dim=2).squeeze(3)
        clustered_v = torch.matmul(v.squeeze(3), S2.transpose(1, 2)).unsqueeze(3)
        side= self.conv1(clustered_q)
        side= side + self.conv2(side)
        side= self.conv3(side)
        side= clustered_q + side
        feature,att = self.transformer(clustered_q.squeeze(dim=3).transpose(1,2), clustered_v.squeeze(dim=3).transpose(1,2))
        feature = feature.unsqueeze(dim=-1).transpose(1,2)
        feature = clustered_q + feature
        fuse=self.aff1(feature,side)
        embed3 = self.conv_group3(q)
        S3 = torch.softmax(embed3, dim=1).squeeze(3)
        fuse = torch.matmul(fuse.squeeze(3), S3).unsqueeze(3)
        out = self.aff2(fuse,q)
        return out


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channels=128, k_num=9, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channels = 4 if self.initial is True else 6
        self.out_channels = out_channels
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.res_module1 = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )
        self.res_module2 = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )

        self.gc = GCET(k_num,out_channels,256)
        self.gsg = GCGT(out_channels,512)
        
        self.linear_0 = nn.Conv2d(out_channels, 1, (1, 1))
        self.linear_1 = nn.Conv2d(out_channels, 1, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):

        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)

        out = self.res_module1(out)
        out = self.gc(out)
        out = self.res_module2(out)
        feature0 = out
        w0 = self.linear_0(out).view(B, -1)
        
        ###
        w0_ds, indices_0 = torch.sort(w0, dim=-1, descending=True)
        indices_0 = indices_0[:, :int(N*0.2)]
        indices_0 = indices_0.view(B, 1, -1, 1)
        scale_kv = torch.gather(out, dim=2, index=indices_0.repeat(1, 128, 1, 1))
        out = self.gsg(out,scale_kv)
        ###
        feature1 = out
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]

            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, feature1_1 = self.down_sampling(x, y, w0, indices, feature1, self.predict)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], feature1_1

class GCTNet(nn.Module):
    def __init__(self, config):
        super(GCTNet, self).__init__()

        self.out_channel = 128

        self.ds_0 = DS_Block(initial=True, predict=False, out_channels=128, k_num=9, sampling_rate=config.sr)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channels=128, k_num=9, sampling_rate=config.sr)

        self.linear = nn.Conv2d(self.out_channel, 2, (1, 1))

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, features_ds = self.ds_1(x_, y1)

        w2 = self.linear(features_ds)

        e_hat = weighted_8points(x2, w2)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat
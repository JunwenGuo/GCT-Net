import torch
import torch.nn.functional as F

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)

    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys

class MatchLoss(object):
    def __init__(self, config):
        self.loss_essential = config.loss_essential
        self.loss_classif = config.loss_classif
        self.ess_loss_margin = config.ess_loss_margin
        self.obj_geod_th = config.obj_geod_th
        self.loss_essential_init_iter = config.loss_essential_init_iter

    def weight_estimation(self, gt_geod_d, is_pos, ones):
        dis = torch.abs(gt_geod_d - self.obj_geod_th) / self.obj_geod_th

        weight_p = torch.exp(-dis)
        weight_p = weight_p*is_pos

        weight_n = ones
        weight_n = weight_n*(1 - is_pos)
        weight = weight_p + weight_n

        return weight

    def run(self, global_step, data, logits, ys, e_hat, y_hat):
        R_in, t_in, xs, pts_virt = data['Rs'], data['ts'], data['xs'], data['virtPts']
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:,:,2:]
        loss = 0
        classif_loss = 0
        # Classification loss
        with torch.no_grad():
            ones = torch.ones((xs.shape[0], 1)).to(xs.device)
        for i in range(len(logits)):
            gt_geod_d = ys[i]
            is_pos = (gt_geod_d < self.obj_geod_th).type(gt_geod_d.type())
            is_neg = (gt_geod_d >= self.obj_geod_th).type(gt_geod_d.type())
            with torch.no_grad():
                pos = torch.sum(is_pos, dim=-1, keepdim=True)
                pos_num = F.relu(pos - 1) + 1
                neg = torch.sum(is_neg, dim=-1, keepdim=True)
                neg_num = F.relu(neg - 1) + 1
                pos_w = neg_num / pos_num
                pos_w = torch.max(pos_w, ones)
                weight = self.weight_estimation(gt_geod_d, is_pos, ones)
            classif_loss += F.binary_cross_entropy_with_logits(weight * logits[i], is_pos, pos_weight=pos_w)

        geod = batch_episym(pts1_virts, pts2_virts, e_hat[-1])
        e_l = torch.min(geod, self.ess_loss_margin*geod.new_ones(geod.shape))
        essential_loss = e_l.mean()

        # Check global_step and add essential loss
        if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
            loss += self.loss_essential * essential_loss
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss

        return loss, essential_loss.item(), classif_loss.item()
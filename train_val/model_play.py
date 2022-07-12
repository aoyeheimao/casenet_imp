import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from .loss import MultiLLFunction
import sys

sys.path.append("../")

# Local imports
import utils.utils as utils
from utils.utils import AverageMeter
import torch.nn.functional as F


def train(args, train_loader, model, optimizer, epoch, curr_lr, win_feats5,
          win_fusion, viz, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to eval mode to make BN unchanged.
    model.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input for Image CNN.
        img_var = utils.check_gpu(0, img)  # BS X 3 X H X W
        target_var = utils.check_gpu(0, target)  # BS X H X W X NUM_CLASSES

        bs = img.size()[0]

        # BS X NUM_CLASSES X 352 X 352
        score_feats5, fused_feats = model(img_var)

        loss_cal = MultiLLFunction()
        feats5_loss = loss_cal(score_feats5, target_var)/(352 * 352)
        fused_feats_loss = loss_cal(fused_feats, target_var)/(352 * 352)
        loss = fused_feats_loss  # feats5_loss +
        # print("loss!!!!!!!!!!:", loss)

        feats5_losses.update(feats5_loss.clone().detach(), bs)
        fusion_losses.update(fused_feats_loss.clone().detach(), bs)
        total_losses.update(loss.clone().detach(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.11f} ({total_loss.avg:.11f})\n'
                  'lr {learning_rate:.10f}\t\n'.format(epoch,
                                                       i,
                                                       len(train_loader),
                                                       batch_time=batch_time,
                                                       data_time=data_time,
                                                       total_loss=total_losses,
                                                       learning_rate=curr_lr))

            trn_feats5_loss = feats5_loss.clone().cpu().detach().numpy()
            trn_fusion_loss = fused_feats_loss.clone().cpu().detach().numpy()
            viz.line(win=win_feats5,
                     name='train_feats5',
                     update='append',
                     X=np.array([global_step]),
                     Y=np.array([trn_feats5_loss]))
            viz.line(win=win_fusion,
                     name='train_fusion',
                     update='append',
                     X=np.array([global_step]),
                     Y=np.array([trn_fusion_loss]))

        global_step += 1

    return global_step


def validate(args, val_loader, model, epoch, win_feats5, win_fusion, viz,
             global_step, winid):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input for Image CNN.
        img_var = utils.check_gpu(0, img)  # BS X 3 X H X W
        target_var = utils.check_gpu(0, target)  # BS X H X W X NUM_CLASSES

        # print("img_var,target_var .max(): ", img_var.max(), target_var.max())

        bs = img.size()[0]

        score_feats5, fused_feats = model(
            img_var)  # BS X NUM_CLASSES X 352 X 352

        loss_cal = MultiLLFunction()
        feats5_loss = loss_cal(score_feats5, target_var)/(352 * 352)
        fused_feats_loss = loss_cal(fused_feats, target_var)/(352 * 352)
        loss = fused_feats_loss  # feats5_loss +

        feats5_losses.update(feats5_loss.clone().detach(), bs)
        fusion_losses.update(fused_feats_loss.clone().detach(), bs)
        total_losses.update(loss.clone().detach(), bs)

        foreground = np.array(torch.sigmoid(fused_feats[:, 1:2, :, :].clone().cpu().detach()))
        background = np.array(torch.sigmoid(fused_feats[:, 0:1, :, :].clone().cpu().detach()))
        # print("fused_feats, foreground .max(): ", fused_feats.clone().cpu().detach().max(), foreground.max())
        # print("fused_feats, foreground .min(): ", fused_feats.clone().cpu().detach().min(), foreground.min())
        # print("fused_feats, foreground .mean(): ", fused_feats.clone().cpu().detach().mean(), foreground.mean())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\n'.
                  format(epoch,
                         i,
                         len(val_loader),
                         batch_time=batch_time,
                         data_time=data_time,
                         total_loss=total_losses))

            trn_feats5_loss = feats5_loss.clone().cpu().detach().numpy()
            trn_fusion_loss = fused_feats_loss.clone().cpu().detach().numpy()
            viz.line(win=win_feats5,
                     name='val_feats5',
                     update='append',
                     X=np.array([global_step]),
                     Y=np.array([trn_feats5_loss]))
            # X=np.array([global_step]),
            # Y=np.array([feats5_losses.avg.cpu()]))
            viz.line(win=win_fusion,
                     name='val_fusion',
                     update='append',
                     X=np.array([global_step]),
                     Y=np.array([trn_fusion_loss]))
            # X=np.array([global_step]),
            # Y=np.array([fusion_losses.avg.cpu()]))
            img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)

            # viz.images(
            #     np.concatenate(
            #         (
            #             img_var.mul_(img_std.type_as(img_var)).add_(img_mean.type_as(img_var)).data.cpu().numpy(),
            #             np.concatenate((foreground.repeat(-1,3,-1,-1), background.repeat(-1,3,-1,-1)), 0),
            #             np.concatenate((np.array(target_var[:, 1:2, :, :].repeat(-1,3,-1,-1).clone().cpu().detach().numpy()),
            #                             np.array(target_var[:, 0:1, :, :].repeat(-1,3,-1,-1).clone().cpu().detach().numpy())), 0),
            #         ),
            #         axis=0
            #     ),
            #     # np.array(F.sigmoid(fused_feats[:, 0:1, :, :]).clone().cpu().data.numpy()),
            #     # target_var[:, 0:1, :, :].clone().cpu().data.numpy()),
            #     # rgb也是灰度图
            #     opts=dict(title='input_rgb, result_foreground, target', caption=''),
            #     nrow=2,
            #     win=9999
            # )
            # images = viz.images(np.random.randn(1, 1, 1, 1),opts=dict(title='Random images', caption='How random.', nrow=5))

            # updaimgwindow = viz.text('rgb image input')
            winid["rgb"] = viz.images(
                img_var.mul_(img_std.type_as(img_var)).add_(img_mean.type_as(img_var)).data.cpu().numpy()
                               # np.array(F.sigmoid(fused_feats[:, 0:1, :, :]).clone().cpu().data.numpy()),
                               # target_var[:, 0:1, :, :].clone().cpu().data.numpy()),
                               ,  # rgb也是灰度图
                opts=dict(title='input_rgb', caption='input_rgb'),
                nrow=2,
                win=winid["rgb"] ,
            )
            # updaedgewindow = viz.text('edge output')
            winid["pre"] = viz.images(
                np.concatenate((foreground, background), 0),  # 生成一张图
                opts=dict(title='result_foreground', caption='result_foreground'),
                nrow=2,
                win=winid["pre"],
            )
            # updatargetwindow = viz.text('edge target')
            winid["gt"] = viz.images(
                np.concatenate((np.array(target_var[:, 1:2, :, :].clone().cpu().detach().numpy()), np.array(target_var[:, 0:1, :, :].clone().cpu().detach().numpy())), 0),  # 生成一张图
                opts=dict(title='target', caption='target'),
                nrow=2,
                win=winid["gt"],
            )

    return fusion_losses.avg, winid

# def WeightedMultiLabelSigmoidLoss(model_output, target):
#     """
#     model_output: BS X NUM_CLASSES X H X W
#     target: BS X H X W X NUM_CLASSES
#     """
#     # Calculate weight. (edge pixel and non-edge pixel)
#     weight_sum = utils.check_gpu(
#         0,
#         target.sum(dim=1).sum(dim=1).sum(dim=1).float().data)  # BS
#     edge_weight = utils.check_gpu(
#         0, weight_sum.data / float(target.size()[1] * target.size()[2]))
#     non_edge_weight = utils.check_gpu(
#         0, (target.size()[1] * target.size()[2] - weight_sum.data) /
#         float(target.size()[1] * target.size()[2]))
#     one_sigmoid_out = F.sigmoid(model_output)
#     zero_sigmoid_out = 1 - one_sigmoid_out
#     target = target.transpose(1, 3).transpose(
#         2, 3).float()  # BS X NUM_CLASSES X H X W
#     loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*target*torch.log(one_sigmoid_out.clamp(min=1e-10)) - \
#             edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*(1-target)*torch.log(zero_sigmoid_out.clamp(min=1e-10))

#     return loss.mean(dim=0).sum()

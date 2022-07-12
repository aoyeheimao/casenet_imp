import os
import sys
import argparse

import numpy as np
from PIL import Image
import cv2
import glob

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CaseNet import build_CaseNet
from prep_dataset.prep_SBD_dataset import RGB2BGR
from prep_dataset.prep_SBD_dataset import ToTorchFormatTensor

import utils.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help="path to the caffemodel containing the trained weights")
    parser.add_argument('-l',
                        '--image_list',
                        type=str,
                        default='',
                        help="list of image files to be tested")
    parser.add_argument('-f',
                        '--image_file',
                        type=str,
                        default='',
                        help="a single image file to be tested")
    parser.add_argument(
        '-d',
        '--image_dir',
        type=str,
        default='',
        help=
        "root folder of the image files in the list or the single image file")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='.',
                        help="folder to store the test results")
    args = parser.parse_args(sys.argv[1:])

    # load input path
    if os.path.exists(args.image_list):
        test_lst = glob.glob(os.path.join(args.image_list, "*.jpg"))
        # with open(args.image_list) as f:
        #     ori_test_lst = [x.strip().split()[0] for x in f.readlines()]
        #     if args.image_dir != '':
        #         test_lst = [
        #             args.image_dir +
        #             x if os.path.isabs(x) else os.path.join(args.image_dir, x)
        #             for x in ori_test_lst
        #         ]
    else:
        image_file = os.path.join(args.image_dir, args.image_file)
        if os.path.exists(image_file):
            ori_test_list = [args.image_file]
            test_lst = [image_file]
        else:
            raise IOError('nothing to be tested!')

    # load net
    num_cls = 2
    model = build_CaseNet(pretrained=True, num_classes=num_cls)
    model = model.cuda()
    model = model.eval()
    cudnn.benchmark = True
    utils.load_pretrained_model(model, args.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        for cls_idx in range(num_cls):
            dir_path = os.path.join(args.output_dir, str(cls_idx))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    # Define normalization for data
    # normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675],
    #                                  std=[1, 1, 1])
    input_size = (352, 352)
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(input_size, Image.BILINEAR),
        # transforms.RandomResizedCrop(input_size[0],
        #                              scale=(0.75, 1.0),
        #                              ratio=(0.75, 1.0)),
        ToTorchFormatTensor(div=False),
        # normalize,
    ])
    for idx_img in range(len(test_lst)):
        img = Image.open(test_lst[idx_img])
        processed_img = img_transform(img).unsqueeze(0)  # 1 X 3 X H X W
        width = processed_img.size()[2]
        height = processed_img.size()[3]
        processed_img_var = utils.check_gpu(0, processed_img)
        score_feats5, score_fuse_feats = model(
            processed_img_var / 255.)  # 1 X 20 X CROP_SIZE X CROP_SIZE

        score_output = score_fuse_feats.permute(
            0, 2, 3, 1).squeeze(0)[:height, :width, :]  # H X W X 20
        for cls_idx in range(num_cls):
            # Convert binary prediction to uint8
            im_arr = np.empty((height, width), np.uint8)
            im_arr = (score_output[:, :, cls_idx].data.cpu().numpy()) * 255.0

            # Store value into img
            img_base_name_noext = os.path.splitext(
                os.path.basename(test_lst[idx_img]))[0]
            if not os.path.exists(os.path.join(args.output_dir, str(cls_idx))):
                os.makedirs(os.path.join(args.output_dir, str(cls_idx)))
            cv2.imwrite(
                os.path.join(args.output_dir, str(cls_idx),
                             img_base_name_noext + '.png'), im_arr)
            print('processed: ' + test_lst[idx_img])

    print('Done!')

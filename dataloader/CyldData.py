import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
import math, random
import torchvision.transforms.functional as TF
import torchvision.transforms as T

seed = 7240
# Minimize randomness
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# class RandomResizedCrop(object):
#     """Crop the given PIL Image to random size and aspect ratio.
#
#     A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
#     is finally resized to given size.
#     This is popularly used to train the Inception networks.
#
#     Args:
#         size: expected output size of each edge
#         scale: range of size of the origin size cropped
#         ratio: range of aspect ratio of the origin aspect ratio cropped
#         interpolation: Default: PIL.Image.BILINEAR
#     """
#     def __init__(self,
#                  size,
#                  scale=(0.08, 1.0),
#                  ratio=(3. / 4., 4. / 3.),
#                  interpolation=Image.BILINEAR):
#         self.size = (size, size)
#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio
#
#     @staticmethod
#     def get_params(img, scale, ratio):
#         """Get parameters for ``crop`` for a random sized crop.
#
#         Args:
#             img (PIL Image): Image to be cropped.
#             scale (tuple): range of size of the origin size cropped
#             ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
#
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         for attempt in range(10):
#             area = img.size[0] * img.size[1]
#             target_area = random.uniform(*scale) * area
#             aspect_ratio = random.uniform(*ratio)
#
#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))
#
#             if random.random() < 0.5:
#                 w, h = h, w
#
#             if w <= img.size[0] and h <= img.size[1]:
#                 i = random.randint(0, img.size[1] - h)
#                 j = random.randint(0, img.size[0] - w)
#                 return i, j, h, w
#
#         # Fallback
#         w = min(img.size[0], img.size[1])
#         i = (img.size[1] - w) // 2
#         j = (img.size[0] - w) // 2
#         return i, j, w, w
#
#     def set_random(self, img):
#         self.i, self.j, self.h, self.w = self.get_params(
#             img, self.scale, self.ratio)
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be flipped.
#
#         Returns:
#             PIL Image: Randomly cropped and resize image.
#         """
#         if self.i is None:
#             self.set_random(img)
#         return TF.resized_crop(img, self.i, self.j, self.h, self.w, self.size,
#                               self.interpolation)
#
#


input_size = (482,642)  # (352, 352)
crop_size = (352, 352)
class CyldData(data.Dataset):
    def __init__(self, img_folder, edge_folder):
        self.img_folder = img_folder
        self.edge_folder = edge_folder
        self._background_img_files, self._foreground_img_files = [], []
        backgrounds = glob.glob(
            os.path.join(self.edge_folder, "background", "*.png"))
        foregrounds = glob.glob(
            os.path.join(self.edge_folder, "foreground", "*.png"))

        for fore_filename, back_filename in zip(np.sort(foregrounds),
                                                np.sort(backgrounds)):
            self._foreground_img_files.append(fore_filename)
            self._background_img_files.append(back_filename)

        self.height, self.width = input_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self._background_img_files)

    def __getitem__(self, index):
        # cr = RandomResizedCrop(input_size[0],
        #                        scale=(0.5, 1.0),
        #                        ratio=(0.5, 1.0))

        f = Image.open(self._foreground_img_files[index])
        b = Image.open(self._background_img_files[index])
        # f = np.array(cv2.imread(self._foreground_img_files[index], 2))
        # b = np.array(cv2.imread(self._background_img_files[index], 2))
        # f = Image.fromarray(f.astype('float32'), mode='F')
        # b = Image.fromarray(b.astype('float32'), mode='F')
        img_name = os.path.split(self._foreground_img_files[index])[-1]
        # rgb = Image.open(os.path.join(self.img_folder, img_name))
        # rgb = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, img_name), 0), cv2.COLOR_GRAY2RGB))  # gray
        rgb = np.array(cv2.imread(os.path.join(self.img_folder, img_name)))
        rgb = Image.fromarray(rgb, mode='RGB')


        if True:
            _scale = np.random.uniform(0.5, 1.2)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-90, 90.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                f = TF.hflip(f)
                b = TF.hflip(b)

             # 尝试把rotate放到crop后面，防止先旋转导致的边框裁切在缩放的时候丢失边缘
            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)  # Argument resample is deprecated and will be removed since v0.10.0. Used interpolation instead
            #  UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
            f = TF.rotate(f, angle=degree, resample=Image.NEAREST, fill=(0,))
            b = TF.rotate(b, angle=degree, resample=Image.NEAREST, fill=(0,))  # bug appear when using torchvision==0.5.0, by adding fill=(0,) fix it.

            t_rgb = T.Compose([
                T.Resize(scale),  # , interpolation=T.InterpolationMode.NEAREST),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 灰度图删除标准化
            ])

            t_edge = T.Compose([
                T.Resize(scale),  # , interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop(self.crop_size),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            f = t_edge(f)
            b = t_edge(b)


        # cr.set_random(f)
        #
        # f_arr = np.array(f)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # f_arr = cv2.dilate(f_arr, kernel)
        # f = Image.fromarray(f_arr)
        #
        #
        # f, b = cr(f), cr(b)

        # image_transform = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.Resize(input_size, Image.BILINEAR),
        #     transforms.ToTensor()
        # ])
        # edge_transform = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.Resize(input_size, Image.BILINEAR),
        #     transforms.ToTensor()
        # ])

        #
        # f, b = edge_transform(f), edge_transform(b)

        # img_name = self._foreground_img_files[index].split("/")[-1]
        # img_name = os.path.split(self._foreground_img_files[index])[-1]
        # img = Image.open(os.path.join(self.img_folder, img_name))

        # img = cr(img)
        # img = image_transform(img)

        return rgb, torch.cat([b, f], 0)


if __name__ == "__main__":
    img_folder = r"Z:\home\ubuntu\workspace\zhoutianyi\casedge\database\cylinder_train\JPEGImagesClean"
    edge_folder = r"Z:\home\ubuntu\workspace\zhoutianyi\casedge\database\cylinder_train\JPEGImagesClassEdge"
    cda = CyldData(img_folder, edge_folder)
    print(cda[0][0].max(), cda[0][0].min(), cda[0][0].shape)
    print(cda[0][1].max(), cda[0][1].shape)

    # train_loader = torch.utils.data.DataLoader(cda,
    #                                           batch_size=4,
    #                                           shuffle=True,
    #                                           pin_memory=True)
    for i in range(len(cda)):
        img, target = cda[i]
        # target = cda[i][1]
        plt.subplot(131)
        plt.imshow(img.permute(1,2,0))

        plt.subplot(132)
        plt.imshow(target[0, :, :])

        plt.subplot(133)
        plt.imshow(target[1, :, :])
        plt.show()



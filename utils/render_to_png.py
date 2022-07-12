import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import random

seed = 7240
# Minimize randomness
# torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class RENDER():
    def __init__(self):
        self.dir = r"C:\Users\HJH\Downloads"
        self.fold = "dir_000001"

        self.depth = sorted(glob.glob(os.path.join(self.dir, self.fold, 'depth', '*.exr')))
        self.mask = sorted(glob.glob(os.path.join(self.dir, self.fold, 'mask_visib', '*.png')))
        self.rgb = sorted(glob.glob(os.path.join(self.dir, self.fold, 'rgb', '*.png')))

        self.K = [572.4114, 573.57043, 325.2611, 242.04899]

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, idx):
        rgb = np.array(cv2.imread(self.rgb[idx]))
        dep = np.array(cv2.imread(self.depth[idx], 2))
        mask = np.array(cv2.imread(self.mask[idx], 2))

        return {"rgb": rgb, "dep": dep, "mask": mask}

    def instance_base_contour(self, mask):
        cont = np.zeros_like(mask)
        cont = cont[:, :, None].repeat(20, axis=2)
        for i in range(1, mask.max() + 1):
            mask_copy = mask.copy()
            mask_copy[mask != i] = 0
            # cont_tmp = cv2.Laplacian(mask_copy.astype(np.uint8),cv2.CV_64F)
            cont_tmp = cv2.Canny(mask_copy.astype(np.uint8), 0, 1)
            cont[:, :, i - 1] = cont_tmp
        return cont


def visualize_cont(contour):
    color_cont = np.zeros((contour.shape[0], contour.shape[1], 3))
    for i in range(contour.shape[2]):
        color_cont_tmp = np.zeros((contour.shape[0], contour.shape[1], 3))
        color_tmp = np.random.random((3))
        color_cont_tmp[contour[:, :, i] != 0] = color_tmp
        color_cont = color_cont + color_cont_tmp
    color_cont = color_cont / color_cont.max() * 255

    return color_cont

def otsu_canny(image, lowrate=0.5):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        # self.ystart = (inputShape[2] - targetShape[2]) / 2
        # self.xstart = (inputShape[3] - targetShape[3]) / 2
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def cv_edge(frame, net):

    inp = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(640, 480),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    # cv2.imshow("img", out)
    # cv2.imwrite('result.png', out)
    # cv2.waitKey(0)
    return out


if __name__ == "__main__":

    # 加载边缘提取模型
    cv2.dnn_registerLayer('Crop', CropLayer)
    # ! [Register]
    # Load the model.
    cv2.samples.addSamplesDataSearchPath(r"Z:\home\ubuntu\workspace\zhoutianyi\casedge")
    net = cv2.dnn.readNet(cv2.samples.findFile("deploy.prototxt"), cv2.samples.findFile("hed_pretrained_bsds.caffemodel"))
    # 加载数据
    data = RENDER()

    # 存在文件夹判断
    if not os.path.exists("../database/cylinder_train/JPEGImagesClean"):
        os.makedirs("../database/cylinder_train/JPEGImagesClean")

    if not os.path.exists("../database/cylinder_train/JPEGImagesClassEdge/background"):
        os.makedirs("../database/cylinder_train/JPEGImagesClassEdge/background")

    if not os.path.exists("../database/cylinder_train/JPEGImagesClassEdge/foreground"):
        os.makedirs("../database/cylinder_train/JPEGImagesClassEdge/foreground")

    train_index = random.sample(range(len(data)), round(len(data) * 0.9))
    test_index = [item for item in range(len(data)) if item not in train_index]
    for i,index in enumerate(train_index):
        save_name =	"{:0>4d}".format(index) + "_train.png"
        rgb = data[index]["rgb"]
        mask = data[index]["mask"]

        mask_copy = mask.copy()
        # rgb_canny = otsu_canny(rgb)
        net_out = cv_edge(rgb, net)
        rgb_canny = (np.round(net_out-0.25)*255).astype(np.uint8)  # 网络输出处理

        mask_canny = cv2.Canny(mask_copy.astype(np.uint8), 0, 1)  # 前景
        # rgb_canny[rgb_canny == 0] = mask_canny[rgb_canny == 0]
        rgb_canny_background = rgb_canny.copy()

        # 有可能出现图片检测的canny和maskcanny没有对齐，尝试使用膨胀覆盖
        kernel = np.ones((3, 3), np.uint8)
        mask_canny = cv2.dilate(mask_canny, kernel, iterations=1)
        mask_canny_dilate = cv2.dilate(mask_canny, kernel, iterations=5)

        # cv2.imshow("mask_canny_dilate",mask_canny_dilate)
        # cv2.waitKey(0)


        # mask 掉前景的背景
        rgb_canny_background[mask_copy != 0] = 0
        rgb_canny_background[mask_canny_dilate != 0] = 0
        generate_canny = rgb_canny_background + mask_canny  # mask_canny


        # todo: 在rgb和背景周围padding一圈白色边缘，在前景padding一圈黑色边缘（暂定像素3）在dataaugmentaion的时候能够学到更多东西
        mask_canny = np.pad(mask_canny, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        rgb_canny_background = np.pad(rgb_canny_background, ((1, 1), (1, 1)), 'constant', constant_values=(255, 255))


        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(255, 255))
        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))

        generate_canny = rgb_canny_background + mask_canny  # mask_canny


        img = np.concatenate((rgb.astype(np.uint8),
                              # rgb_canny[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              mask_canny[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              rgb_canny_background[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              generate_canny[:, :, None].repeat(3, axis=2).astype(np.uint8)
                              ), axis=1)

        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # exit()


        # # 暂时不需要对每个相同的instance区分边缘
        # contour = data.instance_base_contour(mask)
        # rgb_contour = visualize_cont(contour)
        # img = np.concatenate((rgb.astype(np.uint8), rgb_contour.astype(np.uint8)), axis=1)




        cv2.imwrite(os.path.join("../database/cylinder_train/JPEGImagesClean", save_name),rgb)
        cv2.imwrite(os.path.join("../database/cylinder_train/JPEGImagesClassEdge", "background", save_name), rgb_canny_background)
        cv2.imwrite(os.path.join("../database/cylinder_train/JPEGImagesClassEdge", "foreground", save_name), mask_canny)
        print("{:0>4d}".format(i))

        # cv2.imshow('canny_img', img)
        # cv2.waitKey(0)



    # 存在文件夹判断
    if not os.path.exists("../database/cylinder_test/JPEGImagesClean"):
        os.makedirs("../database/cylinder_test/JPEGImagesClean")

    if not os.path.exists("../database/cylinder_test/JPEGImagesClassEdge/background"):
        os.makedirs("../database/cylinder_test/JPEGImagesClassEdge/background")

    if not os.path.exists("../database/cylinder_test/JPEGImagesClassEdge/foreground"):
        os.makedirs("../database/cylinder_test/JPEGImagesClassEdge/foreground")

    for i,index in enumerate(test_index):
        save_name =	"{:0>4d}".format(index) + "_test.png"
        rgb = data[index]["rgb"]
        mask = data[index]["mask"]

        mask_copy = mask.copy()
        # rgb_canny = otsu_canny(rgb)
        net_out = cv_edge(rgb, net)
        rgb_canny = (np.round(net_out-0.25)*255).astype(np.uint8)  # 网络输出处理

        mask_canny = cv2.Canny(mask_copy.astype(np.uint8), 0, 1)  # 前景
        # rgb_canny[rgb_canny == 0] = mask_canny[rgb_canny == 0]
        rgb_canny_background = rgb_canny.copy()

        # 有可能出现图片检测的canny和maskcanny没有对齐，尝试使用膨胀覆盖
        kernel = np.ones((3, 3), np.uint8)
        mask_canny = cv2.dilate(mask_canny, kernel, iterations=1)
        mask_canny_dilate = cv2.dilate(mask_canny, kernel, iterations=5)

        # cv2.imshow("mask_canny_dilate",mask_canny_dilate)
        # cv2.waitKey(0)


        # mask 掉前景的背景
        rgb_canny_background[mask_copy != 0] = 0
        rgb_canny_background[mask_canny_dilate != 0] = 0
        generate_canny = rgb_canny_background + mask_canny  # mask_canny


        # 在rgb和背景周围padding一圈白色边缘，在前景padding一圈黑色边缘（暂定像素3）在dataaugmentaion的时候能够学到更多东西
        mask_canny = np.pad(mask_canny, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        rgb_canny_background = np.pad(rgb_canny_background, ((1, 1), (1, 1)), 'constant', constant_values=(255, 255))


        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(255, 255))
        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        # rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))

        generate_canny = rgb_canny_background + mask_canny  # mask_canny


        img = np.concatenate((rgb.astype(np.uint8),
                              # rgb_canny[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              mask_canny[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              rgb_canny_background[:, :, None].repeat(3, axis=2).astype(np.uint8),
                              generate_canny[:, :, None].repeat(3, axis=2).astype(np.uint8)
                              ), axis=1)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)


        # # 暂时不需要对每个相同的instance区分边缘
        # contour = data.instance_base_contour(mask)
        # rgb_contour = visualize_cont(contour)
        # img = np.concatenate((rgb.astype(np.uint8), rgb_contour.astype(np.uint8)), axis=1)

        cv2.imwrite(os.path.join("../database/cylinder_test/JPEGImagesClean", save_name),rgb)
        cv2.imwrite(os.path.join("../database/cylinder_test/JPEGImagesClassEdge", "background", save_name), rgb_canny_background)
        cv2.imwrite(os.path.join("../database/cylinder_test/JPEGImagesClassEdge", "foreground", save_name), mask_canny)

        print("test file", "{:0>4d}".format(i))


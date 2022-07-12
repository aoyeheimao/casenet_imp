import cv2
import numpy as np
import os
import tqdm

base_path = '/media/kerwin/a845ad94-53eb-4d8a-8275-8ecc1277b7cb5/cylinder/cylinder/'
out_folder = "JPEGImagesClassEdge"
background_folder = "background"
foreground_folder = "foreground"
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))


def GenerateCategoryEdgeMapping(img_name: str):
    global base_path
    global kernel
    global kernel_large

    name = img_name[6:]
    scene_id = img_name[:6]
    img_id = name.split('.')[0]
    image = np.zeros([1200, 1920])
    mask_dir = os.path.join(base_path, "train_clean", scene_id, "mask_visib")
    for file in os.listdir(mask_dir):
        if file.split('_')[0] == img_id:
            img = cv2.imread(os.path.join(mask_dir, file))
            x_grad = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
            y_grad = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
            img_output = cv2.Canny(x_grad, y_grad, 50, 150)
            img_output = cv2.dilate(img_output, kernel)
            image += img_output
    cv2.imwrite(
        os.path.join(base_path, out_folder, foreground_folder, img_name),
        image)

    imgg = cv2.imread(os.path.join(base_path, 'JPEGImagesEdge', img_name))
    imgg = cv2.cvtColor(imgg, cv2.COLOR_RGB2GRAY)
    image_mask = np.zeros([1200, 1920])
    for file in os.listdir(mask_dir):
        if file.split('_')[0] == img_id:
            img = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, file)),
                               cv2.COLOR_RGB2GRAY)
            image_mask[img == 255] = 255

    imgg_dilate = cv2.dilate(image_mask, kernel_large)
    imgg[imgg_dilate == 255] = 0
    cv2.imwrite(
        os.path.join(base_path, out_folder, background_folder, img_name), imgg)


if __name__ == '__main__':
    for file in tqdm.tqdm(
            os.listdir(os.path.join(base_path, "JPEGImagesEdgeClean"))):
        GenerateCategoryEdgeMapping(file)
    print('Successful!')

import os
import shutil
import cv2

from tqdm import tqdm
import numpy as np


def move_split(split, save_dir="JPEGImages", to_gray=False):
    for scene_id in os.listdir(split):
        try:
            i = int(scene_id)
        except:
            continue
        print(f"process: {scene_id}")
        scene_dir = os.path.join(split, scene_id, "logo_new2")
        for rgb_file in tqdm(os.listdir(scene_dir)):
            if to_gray:
                img = cv2.imread(os.path.join(scene_dir, rgb_file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.tile(gray[:, :, None], (1, 1, 3))
                cv2.imwrite(os.path.join(save_dir, scene_id + rgb_file), img)
            else:
                shutil.copy(os.path.join(scene_dir, rgb_file),
                            os.path.join(save_dir, scene_id + rgb_file))


if __name__ == "__main__":
    root_path = "/Datasets/boxes/boxes"
    split_list = ["train", "test"]
    for split in split_list:
        move_split(os.path.join(root_path, split),
                   os.path.join(root_path, "JPEGNew2LogoImages"), False)

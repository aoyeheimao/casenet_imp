import os
import json
import math
import tqdm
from copy import deepcopy
import cv2
import open3d as o3d
import numpy as np
from skimage import measure
from PIL import Image
from itertools import groupby
from pycocotools import mask
from detectron2.structures.boxes import BoxMode

data_dict = {
    "info": {
        "description": "ICBIN",
        "url": "https://bop.felk.cvut.cz/datasets/",
        "version": "1.0",
        "year": 2021,
        "contributor": "Zhihao Liang",
        "date_created": "2021/03/29"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    # "categories": [
    #     {"supercategory": "component",
    #      "id": 1,
    #      "name": "component"},
    # ]
}

bias = -math.pi
delta = math.pi / 4
categories = []
categories.append({
    "supercategory": "component",
    "id": 0,
    "name": "__background__"
})
category_num = 2
for i in range(1, category_num):
    categories.append({
        "supercategory": "component",
        "id": i,
        "name": f"angle_{i}"
    })

data_dict["categories"] = categories


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value,
            elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def is_empty_renderer_result(rgb, mask, threshold=1):
    a = np.multiply(mask, rgb).reshape(-1)
    b = a[np.nonzero(a)]
    return b.std() > threshold, b.std()


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def rgbd_to_pcd(img, depth, camera_K, m2mm=True):
    fx = camera_K[0, 0]
    cx = camera_K[0, 2]
    fy = camera_K[1, 1]
    cy = camera_K[1, 2]
    depth_array = depth.astype(np.float32)
    imgH, imgW = depth_array.shape
    color_raw = o3d.geometry.Image(img)
    depth_raw = o3d.geometry.Image(depth_array)
    camera_param = o3d.camera.PinholeCameraIntrinsic(width=imgW,
                                                     height=imgH,
                                                     fx=fx,
                                                     fy=fy,
                                                     cx=cx,
                                                     cy=cy)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    scene_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, camera_param)
    if m2mm:
        scene_o3d.points = o3d.utility.Vector3dVector(
            np.asarray(scene_o3d.points) * 1000.)
    return scene_o3d


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask,
                                pad_width=1,
                                mode='constant',
                                constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1, dtype=object)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def rotation_matrix_to_euler_angles(r):
    r = np.array(r).reshape(3, 3)
    sy = math.sqrt(r[0, 0]**2 + r[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0
    return np.array([x, y, z])


if __name__ == "__main__":
    dataset_name = "boxes"
    data_root = f"/Datasets/{dataset_name}/{dataset_name}/"
    img_id = 0
    anno_id = 0
    anno_folder = "annotations_one_class"
    rot_bbox = False

    os.makedirs(os.path.join(data_root, anno_folder), exist_ok=True)
    split_list = ["train", "test"]
    for split in split_list:
        split_dir = os.path.join(data_root, split)
        for scene_id in os.listdir(split_dir):
            try:
                i = int(scene_id)
            except:
                continue
            rgb_dir = os.path.join(split_dir, scene_id, "rgb")
            depth_dir = os.path.join(split_dir, scene_id, "depth")
            mask_dir = os.path.join(split_dir, scene_id, "mask")
            vis_mask_dir = os.path.join(split_dir, scene_id, "mask_visib")
            with open(os.path.join(split_dir, scene_id, "scene_gt.json"),
                      'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(split_dir, scene_id, "scene_camera.json"),
                      'r') as f:
                scene_camera = json.load(f)
            print(f"processing: {scene_id}")
            view_num = len(os.listdir(rgb_dir))
            for v_id in tqdm.trange(view_num):
                v_id = str(v_id)
                view_id = f"{v_id:0>6}"
                gt_list = scene_gt[v_id]
                num_gt = len(gt_list)

                filename = scene_id + view_id
                img_dict = {
                    "height": 960,
                    "width": 1280,
                    "file_name": filename + ".jpg",
                    "id": deepcopy(img_id),
                    "camera": scene_camera[v_id]
                }
                data_dict["images"].append(img_dict)
                for g_id in range(0, num_gt):
                    gt_id = f"{g_id:0>6}"
                    mask_file = os.path.join(mask_dir,
                                             f"{view_id}_{gt_id}.png")
                    binary_mask = np.array(Image.open(mask_file))
                    vis_mask_file = os.path.join(vis_mask_dir,
                                                 f"{view_id}_{gt_id}.png")
                    if not os.path.exists(vis_mask_file):
                        print(f"skip invisible instance {view_id}_{gt_id}.")
                    vis_binary_mask = np.array(Image.open(vis_mask_file))
                    vis_binary_mask = (vis_binary_mask / 255).astype(np.int32)
                    binary_mask_encoded = mask.encode(
                        np.asfortranarray(
                            (vis_binary_mask * 255).astype(np.uint8)))
                    area = mask.area(binary_mask_encoded)
                    # skip very small object
                    if area < 400:
                        # print(
                        #     f"skip annotation whose area is {area} from image {view_id}_{gt_id}"
                        # )
                        continue

                    contours, _ = cv2.findContours(
                        vis_binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, 1)
                    contours_rotbbox, _ = cv2.findContours(
                        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, 1)
                    if rot_bbox:
                        rect = cv2.minAreaRect(contours_rotbbox[0])
                        ((cx, cy), (w, h), angle) = rect
                        # rvec, tvec = cv2.Rodrigues(
                        #     np.asarray(gt_list[g_id]["cam_R_m2c"]).reshape(
                        #         3,
                        #         3))[0], np.asarray(gt_list[g_id]["cam_t_m2c"])
                        # points = np.array([[0, 0, 0], [1., 0., 0]])
                        # image_points, _ = cv2.projectPoints(
                        #     points, rvec, tvec,
                        #     np.asarray(
                        #         scene_camera[str(g_id)]["cam_K"]).reshape(
                        #             3, 3), np.zeros(5, ))
                        # image_points_ = image_points.squeeze(1)
                        # angle = np.arctan2(
                        #     image_points_[1, 1] - image_points_[0, 1],
                        #     image_points_[1, 0] -
                        #     image_points_[0, 0]) * 180 / np.pi
                        # if w < h:
                        #     temp = w
                        #     w = h
                        #     h = temp
                    else:
                        bounding_box = mask.toBbox(binary_mask_encoded)
                        cx, cy, w, h = bounding_box
                        cx += w / 2
                        cy += h / 2
                        angle = 0

                    segmentations = []
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        segmentation = contour.ravel().tolist()
                        segmentations.append(segmentation)
                    x_angle, y_angle, z_angle = rotation_matrix_to_euler_angles(
                        gt_list[g_id]["cam_R_m2c"])
                    x_label = np.floor(
                        (x_angle - bias) / delta / 2).astype(np.int32)
                    y_label = np.floor(
                        (y_angle - bias) / delta / 2).astype(np.int32)
                    z_label = np.floor(
                        (z_angle - bias) / delta / 2).astype(np.int32)
                    annotation_info = {
                        "id": deepcopy(anno_id),
                        "image_id": deepcopy(img_id),
                        # "category_id":
                        # int(x_label * 16 + y_label * 4 + z_label + 1),
                        "category_id": 1,
                        "iscrowd": 0,
                        "area": area.tolist(),
                        "bbox": [cx, cy, w, h, -angle],
                        "bbox_mode": BoxMode.XYWHA_ABS,
                        "segmentation": segmentations,
                        "width": binary_mask.shape[1],
                        "height": binary_mask.shape[0],
                    }
                    anno_id += 1
                    data_dict["annotations"].append(annotation_info)
                img_id += 1
        with open(os.path.join(data_root, f"{anno_folder}/{split}_new.json"),
                  'w') as f:
            json.dump(data_dict, f)

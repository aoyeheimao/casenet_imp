import torch
import torch.utils.data

from dataloader.CyldData import CyldData


def get_dataloader(args):
    img_folder = "./database/cylinder_train/JPEGImagesClean"  # "/Datasets/cylinder/cylinder/JPEGImagesClean"
    edge_folder = "./database/cylinder_train/JPEGImagesClassEdge"  # "/Datasets/cylinder/cylinder/JPEGImagesClassEdge"
    train_dataset = CyldData(img_folder, edge_folder)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    img_folder = "./database/cylinder_test/JPEGImagesClean"  # "/Datasets/cylinder/cylinder/JPEGImagesClean"
    edge_folder = "./database/cylinder_test/JPEGImagesClassEdge"  # "/Datasets/cylinder/cylinder/JPEGImagesClassEdge"
    val_dataset = CyldData(img_folder, edge_folder)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    return train_loader, val_loader

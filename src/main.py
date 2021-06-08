import os
import argparse
from datetime import datetime
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("../detr")
from engine import evaluate, train_one_epoch
from models import build_model
import util.misc as utils
import datasets.transforms as R

from config import Args
from table_datasets import PDFTablesDataset, TightAnnotationCrop, RandomPercentageCrop, RandomErasingWithTarget, ToPILImageWithTarget, RandomMaxResize, RandomCrop


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root_dir', help="Root data directory for images and labels")
    parser.add_argument('--backbone', default='resnet101', help="Backbone for the model")
    parser.add_argument('--data_type', choices=['detection', 'structure'], default='structure',
                        help="toggle between structure recognition and table detection")

    return parser.parse_args()

def make_structure_coco_transforms(image_set):

    normalize = R.Compose([
        R.ToTensor(),
        R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    random_erasing = R.Compose([
        R.ToTensor(),
        RandomErasingWithTarget(p=0.5, scale=(0.003, 0.03), ratio=(0.1, 0.3), value='random'),
        RandomErasingWithTarget(p=0.5, scale=(0.003, 0.03), ratio=(0.3, 1), value='random'),
        ToPILImageWithTarget()
    ])

    if image_set == 'train':
        return R.Compose([
            RandomCrop(1, 10, 10, 10, 10),
            RandomMaxResize(900, 1100),
            random_erasing,
            normalize
        ])

    if image_set == 'val':
        return R.Compose([
            RandomMaxResize(1000, 1000),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')
    
def make_detection_coco_transforms(image_set):

    normalize = R.Compose([
        R.ToTensor(),
        R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return R.Compose([
            R.RandomSelect(
                TightAnnotationCrop([0, 1], 100, 150, 100, 150),
                RandomPercentageCrop(1, 0.1, 0.1, 0.1, 0.1),
                p=0.2
            ),
            RandomMaxResize(704, 896),
            normalize
        ])
    
    if image_set == 'val':
        return R.Compose([
            RandomMaxResize(800, 800),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')
    
def get_transform(data_type, image_set):
    if data_type == 'structure':
        return make_structure_coco_transforms(image_set)
    else:
        return make_detection_coco_transforms(image_set)
    
def get_class_list_set(data_type):
    if data_type == 'structure':
        class_map = {'table': 0, 'table column': 1, 'table row': 2, 'table column header': 3, 'table projected row header': 4, 'table spanning cell': 5, 'no object': 6}
        class_list = list(class_map)
        class_set = set(class_map.values())#.remove('no object')
        class_set.remove(class_map['no object'])
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
        class_list = list(class_map)
        class_set = set(class_map.values())#.remove('no object')
        class_set.remove(class_map['no object'])
    return class_map, class_list, class_set
    
def get_dataloaders(args):
    # Datasets
    class_map, class_list, class_set = get_class_list_set(args.data_type)

    data_parent_directory = args.data_root_dir
    print("loading train")
    dataset_train = PDFTablesDataset(os.path.join(data_parent_directory, "train"), get_transform(args.data_type, "train"),
                                 do_crop=False, max_neg=0, make_coco=False,image_extension=".jpg",
                                 xml_fileset="100_objects_filelist.txt", class_list=class_list, class_set=class_set, class_map=class_map)
    print("loading val")
    dataset_val = PDFTablesDataset(os.path.join(data_parent_directory, "val"), get_transform(args.data_type, "val"),
                               do_crop=False, make_coco=True, image_extension=".jpg",
                              xml_fileset="100_objects_filelist.txt", class_list=class_list, class_set=class_set, class_map=class_map)
    

    print("loading dataloaders")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 2*args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_train, data_loader_val, dataset_val, len(dataset_train)

def main():
    cmd_args = get_args().__dict__
    args = Args
    for key in cmd_args:
        val = cmd_args[key]
        setattr(args, key, val)
    print(args.__dict__)

    device = torch.device(args.device)

    # Paths
    run_date = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = os.path.join(args.data_root_dir, "output", run_date)
    print("Output directory: ", output_directory)
    model_save_path = os.path.join(output_directory, 'model.pth')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    print("loading data")
    dataloading_time = datetime.now()
    data_loader_train, data_loader_val, dataset_val, train_len = get_dataloaders(args)
    print("finished loading data in :", datetime.now()-dataloading_time)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model_without_ddp = model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_gamma)

    max_batches_per_epoch = int(train_len / args.batch_size)
    print("Max batches per epoch: {}".format(max_batches_per_epoch))
    
    print("Start training")
    start_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        print('-'*100)

        epoch_timing = datetime.now()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, max_batches_per_epoch=max_batches_per_epoch, print_freq=10)
        print("Epoch completed in ", datetime.now() - epoch_timing)

        lr_scheduler.step()
            
        pubmed_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, dataset_val, device, None)
        print("pubmed: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f}".format(pubmed_stats['coco_eval_bbox'][1],
                                                                          pubmed_stats['coco_eval_bbox'][2],
                                                                          pubmed_stats['coco_eval_bbox'][0],
                                                                          pubmed_stats['coco_eval_bbox'][8]))
        
        torch.save(model.state_dict(), model_save_path)
    
    print('Total training time: ', datetime.now() - start_time)


if __name__ == "__main__":
    main()

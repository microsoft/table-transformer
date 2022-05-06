"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import json
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

import table_datasets as TD
from table_datasets import PDFTablesDataset
from eval import eval_coco, eval_tsr


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir',
                        required=True,
                        help="Root data directory for images and labels")
    parser.add_argument('--config_file',
                        required=True,
                        help="Filepath to the config containing the args")
    parser.add_argument('--backbone',
                        default='resnet18',
                        help="Backbone for the model")
    parser.add_argument(
        '--data_type',
        choices=['detection', 'structure'],
        default='structure',
        help="toggle between structure recognition and table detection")
    parser.add_argument('--model_load_path', help="The path to trained model")
    parser.add_argument('--metrics_save_filepath',
                        help='Filepath to save grits outputs',
                        default='')
    parser.add_argument('--table_words_dir',
                        help="Folder containg the bboxes of table words")
    parser.add_argument('--mode',
                        choices=['train', 'eval', 'grits', 'grits-all'],
                        default='train',
                        help="Toggle between different modes")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--lr_gamma', default=0.9, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoint_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--train_max_size', type=int)
    parser.add_argument('--val_max_size', type=int)
    parser.add_argument('--test_max_size', type=int)
    parser.add_argument('--eval_pool_size', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=1)

    return parser.parse_args()


def get_transform(data_type, image_set):
    if data_type == 'structure':
        return TD.get_structure_transform(image_set)
    else:
        return TD.get_detection_transform(image_set)


def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training, 
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")
    class_map = get_class_map(args.data_type)

    if args.mode == "train":
        dataset_train = PDFTablesDataset(
            os.path.join(args.data_root_dir, "train"),
            get_transform(args.data_type, "train"),
            do_crop=False,
            max_size=args.train_max_size,
            include_eval=False,
            max_neg=0,
            make_coco=False,
            image_extension=".jpg",
            xml_fileset="train_filelist.txt",
            class_map=class_map)
        dataset_val = PDFTablesDataset(os.path.join(args.data_root_dir, "val"),
                                       get_transform(args.data_type, "val"),
                                       do_crop=False,
                                       max_size=args.val_max_size,
                                       include_eval=False,
                                       make_coco=True,
                                       image_extension=".jpg",
                                       xml_fileset="val_filelist.txt",
                                       class_map=class_map)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                            args.batch_size,
                                                            drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn,
                                       num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val,
                                     2 * args.batch_size,
                                     sampler=sampler_val,
                                     drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers)
        return data_loader_train, data_loader_val, dataset_val, len(
            dataset_train)

    elif args.mode == "eval":

        dataset_test = PDFTablesDataset(os.path.join(args.data_root_dir,
                                                     "test"),
                                        get_transform(args.data_type, "val"),
                                        do_crop=False,
                                        max_size=args.test_max_size,
                                        make_coco=True,
                                        include_eval=True,
                                        image_extension=".jpg",
                                        xml_fileset="test_filelist.txt",
                                        class_map=class_map)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = DataLoader(dataset_test,
                                      2 * args.batch_size,
                                      sampler=sampler_test,
                                      drop_last=False,
                                      collate_fn=utils.collate_fn,
                                      num_workers=args.num_workers)
        return data_loader_test, dataset_test

    elif args.mode == "grits" or args.mode == "grits-all":
        dataset_test = PDFTablesDataset(os.path.join(args.data_root_dir,
                                                     "test"),
                                        RandomMaxResize(1000, 1000),
                                        include_original=True,
                                        max_size=args.max_test_size,
                                        make_coco=False,
                                        image_extension=".jpg",
                                        xml_fileset="test_filelist.txt",
                                        class_map=class_map)
        return dataset_test


def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path,
                                       map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


def train(args, model, criterion, postprocessors, device):
    """
    Training loop
    """
    # Paths
    run_date = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = os.path.join(args.data_root_dir, "output", run_date)
    if args.model_load_path:
        output_directory = os.path.split(args.model_load_path)[0]
    print("Output directory: ", output_directory)
    model_save_path = os.path.join(output_directory, 'model.pth')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("loading data")
    dataloading_time = datetime.now()
    data_loader_train, data_loader_val, dataset_val, train_len = get_data(args)
    print("finished loading data in :", datetime.now() - dataloading_time)

    model_without_ddp = model
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_drop,
                                                   gamma=args.lr_gamma)

    max_batches_per_epoch = int(train_len / args.batch_size)
    print("Max batches per epoch: {}".format(max_batches_per_epoch))

    if args.model_load_path:
        checkpoint = torch.load(args.model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        print('-' * 100)

        epoch_timing = datetime.now()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            max_batches_per_epoch=max_batches_per_epoch,
            print_freq=1000)
        print("Epoch completed in ", datetime.now() - epoch_timing)

        lr_scheduler.step()

        pubmed_stats, coco_evaluator = evaluate(model, criterion,
                                                postprocessors,
                                                data_loader_val, dataset_val,
                                                device, None)
        print("pubmed: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f}".
              format(pubmed_stats['coco_eval_bbox'][1],
                     pubmed_stats['coco_eval_bbox'][2],
                     pubmed_stats['coco_eval_bbox'][0],
                     pubmed_stats['coco_eval_bbox'][8]))

        # Save current model training progress
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_save_path)

        # Save checkpoint for evaluation
        if (epoch+1) % args.checkpoint_freq == 0:
            model_save_path_epoch = os.path.join(output_directory, 'model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), model_save_path_epoch)

    print('Total training time: ', datetime.now() - start_time)


def main():
    cmd_args = get_args().__dict__
    config_args = json.load(open(cmd_args['config_file'], 'rb'))
    config_args.update(cmd_args)
    args = type('Args', (object,), config_args)
    print(args.__dict__)
    print('-' * 100)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("loading model")
    device = torch.device(args.device)
    model, criterion, postprocessors = get_model(args, device)

    if args.mode == "train":
        train(args, model, criterion, postprocessors, device)
    elif args.mode == "eval":
        data_loader_test, dataset_test = get_data(args)
        eval_coco(args, model, criterion, postprocessors, data_loader_test, dataset_test, device)
    elif args.mode == "grits" or args.mode == 'grits-all':
        assert args.data_type == "structure", "GriTS is only applicable to structure recognition"
        dataset_test = get_data(args)
        eval_tsr(args, model, dataset_test, device)


if __name__ == "__main__":
    main()

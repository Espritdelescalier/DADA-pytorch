'''
Training script for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''
from __future__ import print_function

import os
import shutil
import time
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import cifar10_data
from tensorboardX import SummaryWriter
from config import Config
from Nets import _G, _D, Train, _G3, _G4
from ViT_custom_local544444_256_rp_noise import Generator, Discriminator
from perceiver_pytorch import Perceiver
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from keras.preprocessing.image import ImageDataGenerator
import cfg
from decomptracedataset import DecompTraceDataset
from dataset_config import DatasetConfig
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from torchmetrics.classification import F1Score


def gen_minibatches(x, y, batch_size, shuffle=False):
    assert len(x) == len(y), "Training data size don't match"
    if shuffle:
        ids = np.random.permutation(len(x))
    else:
        ids = np.arange(len(x))
    for start_idx in range(0, len(x)-batch_size+1, batch_size):
        ii = ids[start_idx:start_idx+batch_size]
        yield x[ii], y[ii]


def setup_dataflow(dataset, train_idx, trainbsz, val_idx, valbsz, workers):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=trainbsz,
        sampler=train_sampler,
        num_workers=workers)

    val_loader = DataLoader(
        dataset,
        batch_size=valbsz,
        sampler=val_sampler,
        num_workers=workers
    )
    return train_loader, val_loader


def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def main():

    args = cfg.parse_args()

    def weights_init_D(m):
        classname = m.__class__.__name__
        if classname.find('Conv1') != -1 or classname.find('ConvTranspose1d') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            # nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.05)
            # nn.init.constant_(m.bias.data, 0)

    def weights_init_G(m):
        classname = m.__class__.__name__
        """if classname.find('Conv2') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        # nn.init.constant_(m.bias.data, 0)"""
        if classname.find('Conv1d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    now = datetime.now()
    current = now.strftime("%Y%m%d%H%M")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(log_dir=args.logs)

    # Random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    aug_params = dict(data_format='channels_first', rotation_range=20.,
                      width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # data Aug
    if args.aug:
        datagen = ImageDataGenerator(**aug_params)
    else:
        datagen = ImageDataGenerator(data_format='channels_first')

    dataset_config = DatasetConfig()
    print(f"dataset config debug print{dataset_config}")

    seq_dataset = DecompTraceDataset(
        root_dir=dataset_config.root_dir,  # "../../../../raid/damien/new_code_decomp/",
        target_length=16384,
        hflip=False,
        vflip=False,
        flip_rate=0.9,
        randshift=True,
        seed=args.seed,
        sep_tok=False,
        normalize=False,
        num_classes=4,
        flat=True,
        one_hot=False
    )

    ros = RandomOverSampler(sampling_strategy="auto")

    dataset_indices = np.arange(len(seq_dataset))
    dataset_y = np.array(seq_dataset.all_labels)

    if args.stratified_split:
        splits = StratifiedShuffleSplit(n_splits=2,
                                        train_size=0.85, random_state=args.seed)
        train_y, test_y = splits.split(dataset_indices, dataset_y)
    else:
        train_y, test_y = train_test_split(
            dataset_indices, train_size=0.85, random_state=args.seed, stratify=dataset_y)

    train_idx_ros, train_y_ros = ros.fit_resample(
        dataset_indices[train_y].reshape(-1, 1),
        dataset_y[train_y].reshape(-1, 1)
    )

    test_idx_ros, test_y_ros = ros.fit_resample(
        dataset_indices[test_y].reshape(-1, 1),
        dataset_y[test_y].reshape(-1, 1)
    )

    train_loader, test_loader = setup_dataflow(
        seq_dataset,
        train_idx_ros.flatten(),
        args.train_batch_size,
        test_idx_ros.flatten(),
        args.test_batch_size,
        args.num_workers
    )

    """if not os.path.isdir(opt.save_img):
        os.mkdir(opt.save_img)"""
    if not os.path.isdir(args.logs):
        os.mkdir(args.logs)
    # record loss values
    f = open(args.logs+current+"seed_"+str(args.seed)+'_sequence_' +
             str(dataset_config.target_length)+'loss.txt', 'w')
    loss_res = []

    # Data
    # trainx, trainy = cifar10_data.load(opt.data_dir, subset='train')
    # testx, testy = cifar10_data.load(opt.data_dir, subset='test')

    # Model
    G = _G4(num_classes=args.num_classes)
    # D = _D(num_classes=opt.num_classes)
    # G = eval(args.gen_model+'.Generator')(args=args)
    # D = eval(args.dis_model+'.Discriminator')(args=args)
    """G = Generator(
        args=args, num_classes=args.num_classes, logits=False).float()"""
    """D = Discriminator(
        args=args, num_classes=args.num_classes).float()"""

    # For perceiver the input expected is of shape batch, sequence, channels
    D = Perceiver(
        input_channels=1,
        input_axis=1,
        num_freq_bands=6,
        max_freq=10.,
        depth=6,
        num_latents=256,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=args.num_classes*2,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=2
    ).float()

    if use_cuda:
        D = torch.nn.DataParallel(D).cuda()
        G = torch.nn.DataParallel(G).cuda()

        cudnn.benchmark = True
    # D.apply(weights_init_D)
    G.apply(weights_init_G)
    print('    G params: %.2fM,D params: %.2fM' % (sum(p.numel()
          for p in G.parameters())/1000000.0, sum(p.numel() for p in D.parameters())/1000000.0))
    optimizerD = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    scheduler_G = lr_scheduler.ReduceLROnPlateau(
        optimizerG, 'min', threshold=0.001, factor=0.5)
    T = Train(G, D, optimizerG, optimizerD,
              args.num_classes, args.latent_dim)
    # data shffule
    """train = {}
    for i in range(10):
        train[i] = trainx[trainy == i][:opt.count]
    y_data = np.concatenate([trainy[trainy == i][:opt.count]
                            for i in range(10)], axis=0)
    x_data = np.concatenate([train[i] for i in range(10)], axis=0)
    ids = np.arange(x_data.shape[0])
    np.random.shuffle(ids)
    trainx = x_data[ids]
    trainy = y_data[ids]

    datagen.fit(trainx)
    """

    # Train
    best_acc = 0.0
    weight_gen_loss = 0.0
    best_fone = 0.0
    f1_train = F1Score(task='multiclass', num_classes=4).cuda()
    f1_test = F1Score(task='multiclass', num_classes=4).cuda()
    for epoch in range(args.max_epoch):
        D_loss, G_loss, Train_acc = 0.0, 0.0, 0.0
        nr_batches_test = 0
        nr_batches_train = 0
        index = 0
        if epoch == args.G_epochs:
            weight_gen_loss = 1.0
        # train G
        if epoch < args.G_epochs:
            for _, x_batch, _, y_batch, _ in train_loader:
                x_batch = x_batch[:, None, :].float()
                gen_y = torch.from_numpy(np.int32(np.random.choice(
                    args.num_classes, (y_batch.shape[0],)))).long()
                """
                # TODO avoir un gen_y identique aux vrai labels pour aider la feature loss
                """
                d_loss, train_acc, f1_train = T.train_batch_disc(
                    x_batch, y_batch, gen_y, weight_gen_loss, f1_train
                )
                # print(y_batch)
                D_loss += d_loss
                Train_acc += train_acc
                # f_train = f1_train.compute()
                for j in range(2):
                    gen_y_ = y_batch
                    G_loss += T.train_batch_gen(
                        x_batch,
                        gen_y_,
                        weight_gen_loss
                    )
                nr_batches_train += 1
        else:
            # train Classifier

            for _, x_batch, _, y_batch, _ in train_loader:
                x_batch = x_batch[:, None, :].float()
                index += 1
                gen_y = torch.from_numpy(np.int32(np.random.choice(
                    args.num_classes, (y_batch.shape[0],)))).long()
                d_loss, train_acc, f1_train = T.train_batch_disc(
                    x_batch, y_batch, gen_y, weight_gen_loss, f1_train)
                D_loss += d_loss
                Train_acc += train_acc
                nr_batches_train += 1
        f_train = f1_train.compute()
        f1_train.reset()
        """if index == nr_batches_train:
                break"""
        D_loss /= nr_batches_train
        G_loss /= (nr_batches_train*2)
        # scheduler_G.step(G_loss)
        Train_acc /= nr_batches_train

        # test
        test_acc = 0.0
        if epoch > args.G_epochs and epoch % 50 == 0:
            adjust_learning_rate(optimizerD, 0.1)

        for _, x_batch, _, y_batch, _ in test_loader:
            x_batch = x_batch[:, None, :].float()
            # print("x batch", x_batch.shape)
            test_acc_res, f1_test = T.test(x_batch, y_batch, f1_test)
            test_acc += test_acc_res
            nr_batches_test += 1
        f_test = f1_test.compute()
        f1_test.reset()
        test_acc /= nr_batches_test
        # f_test = f1_test.compute()

        if test_acc > best_acc:
            best_acc = test_acc
            # save gen img

        if f_test > best_fone:
            best_fone = f_test
            # save gen img

        if epoch <= args.G_epochs:
            # T.save_png(opt.save_img, epoch)
            T.save_img(args.save_img, epoch, 2048, args.img_size,
                       args.num_classes, args)
            pass

        if (epoch+1) % (args.print_freq) == 0:
            print("Iteration %d, D_loss = %.4f,G_loss = %.4f,train acc = %.4f, train f1 = %.4f, test acc = %.4f, test f1 = %.4f, best acc = %.4f, best f1 = %.4f, lr = %.10f, G_lr = %.10f" % (
                epoch, D_loss, G_loss, Train_acc, f_train, test_acc, f_test, best_acc, best_fone, optimizerD.param_groups[0]['lr'], optimizerG.param_groups[0]['lr']))
            loss_res.append("Iteration %d, D_loss = %.4f,G_loss = %.4f,train acc = %.4f, train f1 = %.4f, test acc = %.4f, test f1 = %.4f, best acc = %.4f, best f1 = %.4f, lr = %.10f, G_lr = %.10f\n" % (
                epoch, D_loss, G_loss, Train_acc, f_train, test_acc, f_test, best_acc, best_fone, optimizerD.param_groups[0]['lr'], optimizerG.param_groups[0]['lr']))
        # viso
        writer.add_scalar('train/D_loss', D_loss, epoch)
        writer.add_scalar('train/G_loss', G_loss, epoch)
        writer.add_scalar('train/acc', Train_acc, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
    f.writelines(loss_res)
    f.close()


if __name__ == '__main__':
    main()

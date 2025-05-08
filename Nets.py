'''
build model for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''
import torch
from nn import MLPConcatLayer, ConvConcatLayer
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as Weight_norm
import numpy as np
import plotting
import matplotlib.pyplot as plt


class CosineLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(CosineLoss, self).__init__()

    def forward(self, input_seq, target):
        # sigm = nn.Sigmoid()
        # target = sigm(target)
        cos = nn.CosineSimilarity(dim=1, eps=1e-7)
        cosine_loss = 1 - cos(input_seq, target)
        return cosine_loss.mean()


class _G(nn.Module):
    def __init__(self, num_classes):  # inputs (batch,c,w,h)
        super(_G, self).__init__()
        self.num = num_classes
        self.mlp = MLPConcatLayer(self.num)
        self.linear = nn.Linear(100+self.num, 64*1024)
        self.concat = ConvConcatLayer(self.num)
        self.main = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(True))

        self.main2 = nn.Sequential(  # batch,512,8,8
            nn.ConvTranspose1d(1024+self.num, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True))

        self.main3 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(1024+self.num, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))

        self.main4 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(512+self.num, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))

        self.main5 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(512+self.num, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))

        self.main6 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(256+self.num, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))

        self.main7 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(256+self.num, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True))

        self.main8 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d(128+self.num, 8, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs, gen_y):
        inputs = self.mlp(inputs, gen_y)  # batch,4*4*512
        inputs = self.linear(inputs)

        inputs = inputs.reshape(inputs.shape[0], 1024, 64)  # batch,512,4,4
        inputs = self.main(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,512+self.num,4,4
        # print(f"input shape {inputs.shape}")
        """inputs = self.main2(inputs)  # batch,256,8,8
        print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,256+self.num,8,8"""
        inputs = self.main3(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,128+self.num,16,16
        """inputs = self.main4(inputs)  # batch,128,16,16
        print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)"""
        inputs = self.main5(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main6(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main7(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        output = self.main8(inputs)  # batch,3,32,32
        # print(f"output shape {output.shape}")
        return output


class _G2(nn.Module):
    def __init__(self, num_classes):  # inputs (batch,c,w,h)
        super(_G2, self).__init__()
        self.num = num_classes
        self.mlp = MLPConcatLayer(self.num)
        self.linear = nn.Linear(100+self.num, 64*1024)
        self.concat = ConvConcatLayer(self.num)
        self.main = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(True))

        self.main2 = nn.Sequential(  # batch,512,8,8
            nn.ConvTranspose1d(1024+self.num, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True))

        self.main3 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(1024+self.num, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))

        self.main4 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(512+self.num, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))

        self.main5 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(512+self.num, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))

        self.main6 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(256+self.num, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))
        self.main7 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d(256+self.num, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True))

        self.main8 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d(128+self.num, 8, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs, gen_y):
        inputs = self.mlp(inputs, gen_y)  # batch,4*4*512
        inputs = self.linear(inputs)

        inputs = inputs.reshape(inputs.shape[0], 1024, 64)  # batch,512,4,4
        inputs = self.main(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,512+self.num,4,4
        # print(f"input shape {inputs.shape}")
        """inputs = self.main2(inputs)  # batch,256,8,8
        print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,256+self.num,8,8"""
        inputs = self.main3(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,128+self.num,16,16
        """inputs = self.main4(inputs)  # batch,128,16,16
        print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)"""
        inputs = self.main5(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main6(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main7(inputs)  # batch,128,16,16
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        output = self.main8(inputs)  # batch,3,32,32
        # print(f"output shape {output.shape}")
        return output


class _Gdiv(nn.Module):
    def __init__(self, num_classes):  # inputs (batch,c,w,h)
        super(_Gdiv, self).__init__()
        self.num = num_classes
        self.mlp = MLPConcatLayer(self.num)
        self.linear = nn.Linear(100+self.num, (64*1024//8))
        self.concat = ConvConcatLayer(self.num)
        self.main = nn.Sequential(
            nn.BatchNorm1d(1024//8),
            nn.ReLU(True))

        """self.main2 = [nn.Sequential(  # batch,512,8,8
            nn.ConvTranspose1d(1024+self.num, 1024/8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)) for x in range(8)]"""

        self.main3 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d((1024//8+self.num), (512//8),
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(512//8),
            nn.ReLU(True))

        self.main4 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d((512//8+self.num), (512//8),
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(512//8),
            nn.ReLU(True))

        self.main5 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d((512//8+self.num), (256//8),
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(256//8),
            nn.ReLU(True))

        self.main6 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d((256//8+self.num), (256//8),
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(256//8),
            nn.ReLU(True))

        self.main7 = nn.Sequential(  # batch,512,16,16
            nn.ConvTranspose1d((256//8+self.num), (128//8),
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(128//8),
            nn.ReLU(True))

        self.main8 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((128//8+self.num), 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs, gen_y):
        inputs = self.mlp(inputs, gen_y)  # batch,4*4*512
        inputs = self.linear(inputs)
        inputs = inputs.reshape(inputs.shape[0], 1024//8, 64)  # batch,512,4,4

        inputs = self.main(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)  # batch,512+self.num,4,4
        inputs = self.main3(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main5(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main6(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        inputs = self.main7(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")
        inputs = self.concat(inputs, gen_y)
        output = self.main8(inputs)  # batch,512,4,4
        # print(f"input shape {inputs.shape}")

        return output


class _G_block(nn.Module):
    def __init__(self, depth_in, out_depth, num_classes):
        super(_G_block, self).__init__()
        self.depth_in = depth_in
        self.out_depth = out_depth
        self.num_classes = num_classes
        self.conv_block = nn.Sequential(
            nn.ConvTranspose1d(self.depth_in+self.num_classes, self.out_depth,
                               4, 2, 1, bias=False),
            nn.BatchNorm1d(self.out_depth),
            nn.ReLU(True)
        )

    def forward(self, inputs):
        output = self.conv_block(inputs)
        return output


class _G_layer(nn.Module):
    def __init__(self, depth_in, out_depth, num_classes):
        super(_G_layer, self).__init__()
        self.depth_in = depth_in
        self.out_depth = out_depth
        self.num = num_classes
        self.g01 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g02 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g03 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g04 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g05 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g06 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g07 = _G_block(self.depth_in, self.out_depth, self.num)
        self.g08 = _G_block(self.depth_in, self.out_depth, self.num)

    def forward(self, inputs):
        output = torch.concat(
            (self.g01(inputs),
             self.g02(inputs),
             self.g03(inputs),
             self.g04(inputs),
             self.g05(inputs),
             self.g06(inputs),
             self.g07(inputs),
             self.g08(inputs)),
            1
        )
        return output


class _G_gen_layer(nn.Module):
    def __init__(self, depth_in, out_depth, num_classes):
        super(_G_gen_layer, self).__init__()
        self.depth_in = depth_in
        self.out_depth = out_depth
        self.num = num_classes
        self.g01 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g02 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g03 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g04 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g05 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g06 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g07 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.g08 = nn.Sequential(  # batch,3,32,32
            nn.ConvTranspose1d((self.depth_in+self.num),
                               self.out_depth, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        output = torch.concat(
            (self.g01(inputs),
             self.g02(inputs),
             self.g03(inputs),
             self.g04(inputs),
             self.g05(inputs),
             self.g06(inputs),
             self.g07(inputs),
             self.g08(inputs)),
            1
        )
        return output


class _G4(nn.Module):
    def __init__(self, num_classes):
        super(_G4, self).__init__()
        self.num = num_classes
        self.mlp = MLPConcatLayer(self.num)
        self.linear = nn.Linear(100+self.num, (64*1024))
        self.concat = ConvConcatLayer(self.num)
        self.main = self.main = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(True))
        self.g1 = _G_layer(1024, 512//8, self.num)
        self.g2 = _G_layer(512, 512//8, self.num)
        self.g3 = _G_layer(512, 256//8, self.num)
        self.g4 = _G_layer(256, 256//8, self.num)
        self.g5 = _G_layer(256, 128//8, self.num)
        self.g6 = _G_gen_layer(128, 1, self.num)

    def forward(self, inputs, gen_y):
        inputs = self.mlp(inputs, gen_y)  # batch,4*4*512
        inputs = self.linear(inputs)
        inputs = inputs.reshape(inputs.shape[0], 1024, 64)
        inputs = self.main(inputs)
        inputs = self.concat(inputs, gen_y)
        inputs = self.g1(inputs)
        inputs = self.concat(inputs, gen_y)
        # inputs = self.g2(inputs)
        # inputs = self.concat(inputs, gen_y)
        inputs = self.g3(inputs)
        inputs = self.concat(inputs, gen_y)
        inputs = self.g4(inputs)
        inputs = self.concat(inputs, gen_y)
        inputs = self.g5(inputs)
        inputs = self.concat(inputs, gen_y)
        output = self.g6(inputs)

        return output


class _G3(nn.Module):
    def __init__(self, num_classes):  # inputs (batch,c,w,h)
        super(_G3, self).__init__()
        self.num = num_classes
        self.g1 = _Gdiv(self.num)
        self.g2 = _Gdiv(self.num)
        self.g3 = _Gdiv(self.num)
        self.g4 = _Gdiv(self.num)
        self.g5 = _Gdiv(self.num)
        self.g6 = _Gdiv(self.num)
        self.g7 = _Gdiv(self.num)
        self.g8 = _Gdiv(self.num)

    def forward(self, inputs, gen_y):
        i1 = self.g1(inputs, gen_y)
        i2 = self.g2(inputs, gen_y)
        i3 = self.g3(inputs, gen_y)
        i4 = self.g4(inputs, gen_y)
        i5 = self.g5(inputs, gen_y)
        i6 = self.g6(inputs, gen_y)
        i7 = self.g7(inputs, gen_y)
        i8 = self.g8(inputs, gen_y)
        # print(i8.shape)
        output = torch.concat((i1, i2, i3, i4, i5, i6, i7, i8), 1)
        # print(output.shape)
        return output


class _D(nn.Module):
    def __init__(self, num_classes):
        super(_D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1, bias=False),  # 32*32
            # nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),  # 32*32
            # nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, 3, 2, 1, bias=False),  # 16*16
            # nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # 16*16
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 192, 3, 1, 1, bias=False),  # 16*16
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 192, 3, 2, 1, bias=False),  # 8*8
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(192, 192, 3, 1, 0, bias=False),  # 6*6
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 192, 1, 1, 0, bias=False),  # 6*6
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.Conv2d(192, 192, 1, 1, 0, bias=False),  # 6*6
            # nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
        )
        self.main2 = nn.Sequential(
            nn.Linear(192, 2*num_classes)
        )

    def forward(self, inputs, feature=False):
        inputs = inputs +\
            torch.normal(mean=torch.zeros((inputs.shape)), std=0.2).cuda()

        output = self.main(inputs)  # 192*6*6
        output = F.adaptive_avg_pool2d(output, [1, 1])  # 192*1*1
        if feature:
            return output
        output = output.view(-1, 192)  # 1*192
        features = self.main2(output)  # 2*10
        return torch.squeeze(features)  # 20


class Train(object):
    def __init__(self, G, D, G_optim, D_optim, num_classes, latent_dim):
        self.G = G
        self.D = D
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.loss = nn.CrossEntropyLoss()
        self.noise = torch.randn(100, 100)
        self.num = num_classes
        self.latent_dim = latent_dim

    def log_sum_exp(self, x, axis=1):
        m = torch.max(x, dim=1)[0]
        return torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))+m

    def train_batch_disc(self, x, y, gen_y, weight_gen_loss, fone):
        noise = torch.randn(y.shape[0], self.latent_dim)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()
            gen_y = gen_y.cuda()
            y = y.cuda()

        self.D.train()
        self.G.train()
        # print(x.shape)
        lab = self.D(x.reshape((y.shape[0], 16384, 1)))
        # print("lab", lab.shape)
        gen = self.G(noise, F.one_hot(gen_y, self.num).float().cuda())
        gen_data = self.D(gen.reshape(x.shape).reshape(
            (y.shape[0], 16384, 1)))  # self.D(gen.reshape(x.shape))

        # (batch,2,self.num) ---< (batch,2)
        source_lab = torch.matmul(torch.reshape(lab, (y.shape[0], 2, self.num)), F.one_hot(
            y, self.num).unsqueeze(-1).float().cuda()).squeeze()
        source_gen = torch.matmul(torch.reshape(gen_data, (gen_y.shape[0], 2, self.num)), F.one_hot(
            gen_y, self.num).unsqueeze(-1).float().cuda()).squeeze()
        # real and fake class loss
        # loss_source_gen = self.loss(source_gen,torch.zeros(gen_y.shape[0]))
        loss_source_lab = torch.mean(self.loss(source_lab, torch.zeros(y.shape[0]).long(
        ).cuda()))+torch.mean(self.loss(source_gen, torch.ones(y.shape[0]).long().cuda()))

        # (batch,2,sel.num) ---<(batch,self.num)
        class_lab = torch.matmul(torch.reshape(lab, (y.shape[0], 2, self.num)).permute(
            0, 2, 1), torch.ones(y.shape[0], 2, 1).float().cuda()).squeeze()  # (batch,self.num)
        class_gen = torch.matmul(torch.reshape(gen_data, (gen_y.shape[0], 2, self.num)).permute(
            0, 2, 1), torch.ones(gen_y.shape[0], 2, 1).float().cuda()).squeeze()
        # real class loss
        loss_class_lab = torch.mean(self.loss(class_lab, y))
        loss_class_gen = torch.mean(self.loss(class_gen, gen_y))

        # total loss
        loss_lab = (1-weight_gen_loss)*loss_source_lab + \
            weight_gen_loss*(loss_class_lab+loss_class_gen)
        # acc
        out_class = torch.topk(class_lab, 1, dim=1).indices.flatten()
        acc = torch.mean((out_class == y).float())
        fone.update(out_class, y)
        self.D_optim.zero_grad()
        loss_lab.backward()
        self.D_optim.step()
        return loss_lab.item(), acc.item(), fone

    def train_batch_gen(self, x, gen_y, weight_gen_loss):
        noise = torch.randn(gen_y.shape[0], self.latent_dim)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()

        self.D.train()
        self.G.train()

        gen = self.G(noise, F.one_hot(gen_y, self.num).float().cuda())
        # print(gen.shape)
        gen_prediction = self.D(gen.reshape(
            x.shape).reshape((x.shape[0], 16384, 1)))

        # self.D(x, True)  # self.D(x, True)
        output_x, feature_lab = self.D(
            x.reshape((x.shape[0], 16384, 1)), return_embeddings=True)
        output_gen, feature_gen = self.D(gen.reshape(
            x.shape).reshape((x.shape[0], 16384, 1)), return_embeddings=True)
        # self.D(gen, True)
        # gen loss
        source_gen = torch.matmul(
            torch.reshape(gen_prediction, (gen_y.shape[0], 2, self.num)),
            F.one_hot(gen_y, self.num).float().cuda().unsqueeze(-1)
        ).squeeze(2)
        loss_source_gen = torch.mean(self.loss(
            source_gen, torch.zeros(gen_y.shape[0]).long().cuda()))
        # feature loss
        m1 = torch.mean(feature_lab, dim=0)
        m2 = torch.mean(feature_gen, dim=0)
        feature_loss = torch.mean(torch.abs(m1-m2))
        # total loss
        total_loss = (1-weight_gen_loss)*(loss_source_gen+0.5*feature_loss)
        """print(
            f"loss source gen {loss_source_gen}, feature loss {feature_loss}, total loss {total_loss}")"""
        self.G_optim.zero_grad()
        self.D_optim.zero_grad()
        total_loss.backward()
        self.G_optim.step()
        return total_loss.item()

    def test(self, x, y, fone):
        self.D.eval()
        self.G.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            output = self.D(x.reshape((x.shape[0], 16384, 1)))
            output = torch.matmul(torch.reshape(output, (y.shape[0], 2, self.num)).permute(
                0, 2, 1), torch.ones(y.shape[0], 2, 1).cuda()).squeeze(2)
            out_class = torch.topk(output, 1, dim=1).indices.flatten()
            acc = torch.mean(
                (out_class == y).float())
            fone.update(out_class, y)
        return acc.item(), fone

    def save_img(self,
                 save_dir,
                 epoch,
                 target_length,
                 sequence_size,
                 num_classes,
                 args):
        """print(
            f"target_length, {target_length}, sequence size, {sequence_size}")"""
        cls_input = torch.from_numpy(
            np.int32(np.repeat(np.arange(num_classes).reshape(num_classes, 1), num_classes, axis=1))).reshape(num_classes*2, 1)
        noise = torch.randn(2*num_classes, self.latent_dim)
        """print(
            f"npise, {noise.shape}, class input, {cls_input.shape}")"""
        if torch.cuda.is_available():
            noise = noise.cuda()
            cls_input = cls_input.long().cuda()
        with torch.no_grad():
            gen_data = self.G(noise, F.one_hot(
                cls_input, num_classes).squeeze().float().cuda())
            gen_data = gen_data.cpu().detach().numpy()

        print("gen data save image ", gen_data.shape)
        fig, ax_num = plt.subplots(gen_data.shape[1])
        for i in range(gen_data.shape[0]):
            # fig, ax_num = plt.subplots(sequence_size//target_length)
            # range(sequence_size//target_length):
            for j in range(gen_data.shape[1]):
                ax_num[j].plot(gen_data.squeeze()[i, j, :])
                # ax_num[j].plot(gen_data.squeeze()[i, j*target_length:(j*target_length)+target_length])
            plt.savefig(save_dir+"gpu"+str(args.gpu_id)+"sequences_example_epoch"+str(epoch) +
                        "_class_"+str(cls_input[i].item())+"_num_"+str(i)+".png")
            for j in range(gen_data.shape[1]):
                ax_num[j].cla()
        plt.close('all')

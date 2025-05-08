self.pos_gen_y_1 = nn.Parameter(
    torch.zeros(1, self.num_classes, embed_dim))
self.pos_gen_y_2 = nn.Parameter(torch.zeros(
    1, self.num_classes, embed_dim))
self.pos_gen_y_3 = nn.Parameter(torch.zeros(
    1, self.num_classes, embed_dim))
self.pos_gen_y_4 = nn.Parameter(torch.zeros(
    1, self.num_classes, embed_dim//(self.factor)))
self.pos_gen_y_5 = nn.Parameter(torch.zeros(
    1, self.num_classes, embed_dim//(self.factor**2)))
self.pos_gen_y_6 = nn.Parameter(torch.zeros(
    1, self.num_classes, embed_dim//(self.factor**3)))

self.pos_gen_y_embed = [
    self.pos_gen_y_1,
    self.pos_gen_y_2,
    self.pos_gen_y_3,
    self.pos_gen_y_4,
    self.pos_gen_y_5,
    self.pos_gen_y_6
]


def forward(self, z, gen_y):
    # print("z", torch.max(z))
    if self.args.latent_norm:
        latent_size = z.size(-1)
        z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
    if self.l2_size == 0:
        x = self.l1(z).view(-1, self.bottom_width, self.embed_dim)
        gen_y = self.lgen_y(gen_y[:, :, None])
    elif self.l2_size > 1000:
        x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size//16)
        # x = torch.concat(self.l2(x), gen_y)
        x = self.l2(x)
    else:
        x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
        x = self.l2(x)
    # print("x after if else", torch.max(x))

    x = torch.concat(
        (x + self.pos_embed[0], gen_y + self.pos_gen_y_embed[0]), dim=1)
    B = x.size()
    H = self.bottom_width
    # print("x before block 1", torch.max(x))
    x = self.blocks_1(x)
    # print("x after block 1", torch.max(x))

    gen_y = x[:, -self.num_classes:, :]
    x, H = bicubic_upsample(x[:, :-self.num_classes, :], H, self.factor)
    # cut embedding of gen_y and concat to x[-self.num_classes:] bellow
    x = torch.concat(
        (x + self.pos_embed[1], gen_y + self.pos_gen_y_embed[1]), dim=1)
    B, _, C = x.size()
    # print("x before block 2", torch.max(x))
    x = self.blocks_2(x)
    # print("x after block 2", torch.max(x))
    # print(torch.max(x))
    gen_y = x[:, -self.num_classes:, :]
    x, H = bicubic_upsample(x[:, :-self.num_classes, :], H, self.factor)
    # cut embedding of gen_y and concat to x[-self.num_classes:] bellow
    x = torch.concat(
        (x + self.pos_embed[2], gen_y + self.pos_gen_y_embed[2]), dim=1)
    B, _, C = x.size()
    x = self.blocks_3(x)
    # print(torch.max(x))
    gen_y = x[:, -self.num_classes:, :]
    x, H = pixel_upsample(x[:, :-self.num_classes, :], H, self.factor)
    # cut embedding of gen_y and concat to x[-self.num_classes:] bellow
    gen_y = _1Ddownsample(gen_y, self.factor)
    gen_y = torch.repeat_interleave(
        gen_y, H//self.window_size, dim=0)
    x = x + self.pos_embed[3]
    B, _, C = x.size()
    x = x.view(B, H, C)
    x = window_partition(x, self.window_size)
    # increase embedding of gen_y and concat to x[-self.num_classes:] bellow
    # TO TRY separate pos embed for gen_y to avoid concat before window_partition
    # x = x.view(-1, self.window_size*self.window_size, C)
    x = torch.concat((x, gen_y + self.pos_gen_y_embed[3]), dim=1)
    x = self.blocks_4(x)
    # print(torch.max(x))
    # x = x.view(-1, self.window_size, self.window_size, C)
    gen_y = x[:, -self.num_classes:, :]
    gen_y = _1Ddownsample(gen_y.permute(2, 1, 0), H //
                          self.window_size).permute(2, 1, 0)
    x = window_reverse(x[:, :-self.num_classes, :],
                       self.window_size, H).view(B, H, C)

    # gen_y = x[:, -self.num_classes:, :]
    x, H = pixel_upsample(x, H, self.factor)
    # cut embedding of gen_y and concat to new x bellow
    gen_y = _1Ddownsample(gen_y, self.factor)
    gen_y = torch.repeat_interleave(
        gen_y, H//self.window_size, dim=0)
    x = x + self.pos_embed[4]
    B, _, C = x.size()
    x = x.view(B, H, C)
    x = window_partition(x, self.window_size)
    # increase embedding of gen_y and concat to new x bellow
    # x = x.view(-1, self.window_size*self.window_size, C)
    x = torch.concat((x, gen_y + self.pos_gen_y_embed[4]), dim=1)
    x = self.blocks_5(x)
    # x = x.view(-1, self.window_size, self.window_size, C)
    gen_y = x[:, -self.num_classes:, :]
    gen_y = _1Ddownsample(gen_y.permute(2, 1, 0), H //
                          self.window_size).permute(2, 1, 0)
    x = window_reverse(x[:, :-self.num_classes, :],
                       self.window_size, H).view(B, H, C)

    # gen_y = x[:, -self.num_classes:, :]
    x, H = pixel_upsample(x, H, self.factor)
    # cut embedding of gen_y and concat to x[-self.num_classes:] bellow
    gen_y = _1Ddownsample(gen_y, self.factor)
    gen_y = torch.repeat_interleave(
        gen_y, H//self.window_size, dim=0)
    x = x + self.pos_embed[5]
    B, _, C = x.size()
    # x = x.view(B, H, W, C)
    x = window_partition(x, self.window_size)
    # increase embedding of gen_y and concat to x[-self.num_classes:] bellow
    # x = x.view(-1, self.window_size*self.window_size, C)
    x = torch.concat((x, gen_y + self.pos_gen_y_embed[5]), dim=1)
    x = self.blocks_6(x)
    gen_y = x[:, -self.num_classes:, :]
    gen_y = _1Ddownsample(gen_y.permute(2, 1, 0), H //
                          self.window_size).permute(2, 1, 0)
    # x = x.view(-1, self.window_size, self.window_size, C)
    x = window_reverse(x[:, :-self.num_classes, :], self.window_size, H).view(
        B, H, C)

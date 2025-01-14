from utils import *
from version import __version__

# minibatch block

# The very original implementation
class MinibatchBlock(nn.Module):
    def __init__(self, in_features, num_kernels, dim_per_kernel=5, minibatch_size=2) -> None:
        super().__init__()
        # this is to sample a minibatch from the entire batch. 
        # The original implementation instead keeps the batch size small to achieve minibatch. 
        # They are equivalent.
        self.minibatch_size = minibatch_size
        
        self.in_features = in_features
        self.num_kernels = num_kernels # out_features = in_features + self.num_kernels
        self.theta = nn.Linear(in_features, num_kernels * dim_per_kernel, bias=False)
        self.log_weight_scale = nn.Parameter(torch.zeros(num_kernels, dim_per_kernel, dtype=torch.float, requires_grad=True))
        
        self.b = nn.Parameter(torch.full((num_kernels,), -1, dtype=torch.float, requires_grad=True))
        
        torch.nn.init.normal_(self.theta.weight, std=0.05)
        
    def get_out_features(self):
        return self.in_features + self.num_kernels
        
    def forward(self, x):
        if len(x.shape) > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            x = x.view(x.shape[0], -1)
            
        _batch_szie = x.shape[0] // self.minibatch_size
            
        # activation.shape = (batch_size, minibatch_size, num_kernels, dim_per_kernel)
        activation = self.theta(x)
        activation = activation.view(x.shape[0]//self.minibatch_size, self.minibatch_size, self.num_kernels, -1)
        abs_dif    = torch.sum(torch.abs(activation.unsqueeze(4) - activation.permute(0,2,3,1).unsqueeze(1)), dim=3)\
            + 1e6 * torch.eye(self.minibatch_size, device=x.device).unsqueeze(1).unsqueeze(0).expand(_batch_szie, -1, -1, -1)
            
        f = torch.sum(torch.exp(-abs_dif), dim=3) + self.b
        f = f.reshape(x.shape[0], -1)
        
        return torch.cat([x, f], dim=1)
  
class MinibatchStd(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        assert group_size > 1, 'it is meaningless to have minibatching when minibatch_size is 1'
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'
    
# communication blocks

class CommConv(nn.Module):
    def __new__(cls, win_size=3, n_dim=2, comm_type='mean'):
        assert comm_type in ['mean', 'sharpen', 'maxpool', 'learned'], "comm_type must be 'mean', 'sharpen', or 'maxpool'."
        assert win_size % 2 == 1, "win_size must be odd."
       
        if n_dim == 2:
            return CommConv3D(win_size, comm_type=comm_type)
        elif n_dim == 3:
            return CommConv4D(win_size, comm_type=comm_type)
        else:
            raise ValueError(f"n_dim must be 2, or 3, but got {n_dim}.")
   
    def forward(self, x):
        raise NotImplementedError
        # return self.conv(
        #     F.pad(x.unsqueeze(0), (self.conv.padding[0],)*2, mode=self.padding_mode)
        #     ).squeeze(0)

class CommConv3D(nn.Module):
    def __init__(self, win_size=3, comm_type='mean') -> None:
        super().__init__()
        self.pad_size = (win_size-1)//2

        if comm_type == 'mean':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1), bias=False)
            self.conv.requires_grad_(False)
            self.conv.weight.data = torch.ones_like(self.conv.weight.data)/win_size
        elif comm_type == 'sharpen':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1), bias=False)
            self.conv.requires_grad_(False)
            self.conv.weight.data =-torch.ones_like(self.conv.weight.data)/win_size
            self.conv.weight.data[:,:,win_size//2] += 1
        elif comm_type == 'maxpool':
            self.conv = nn.MaxPool3d((win_size,1,1), stride=1)
        elif comm_type == 'learned':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1))
        else:
            raise NotImplementedError

        # register this module as a buffer
        if comm_type != 'maxpool':
            self.register_buffer('weight', self.conv.weight.data)

    def forward(self, x):
        # circular padding
        x = torch.cat([x[-self.pad_size:,:,:], x, x[:self.pad_size,:,:]], dim=0)
        return self.conv(x.unsqueeze(0)).squeeze(0)

class CommConv4D(nn.Module):
    def __init__(self, win_size=3, comm_type='mean') -> None:
        super().__init__()
        self.pad_size = (win_size-1)//2

        if comm_type == 'mean':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1), bias=False)
            self.conv.requires_grad_(False)
            self.conv.weight.data = torch.ones_like(self.conv.weight.data)/win_size
        elif comm_type == 'sharpen':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1), bias=False)
            self.conv.requires_grad_(False)
            self.conv.weight.data =-torch.ones_like(self.conv.weight.data)/win_size
            self.conv.weight.data[:,:,win_size//2] += 1
        elif comm_type == 'maxpool':
            self.conv = nn.MaxPool3d((win_size,1,1), stride=1)
        elif comm_type == 'learned':
            self.conv = nn.Conv3d(1, 1, (win_size, 1, 1))
        else:
            raise NotImplementedError

        # register this module as a buffer
        if comm_type != 'maxpool':
            self.register_buffer('weight', self.conv.weight.data)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.view(b,c*h,w)
        # circular padding
        x = torch.cat([x[-self.pad_size:,:], x, x[:self.pad_size,:]], dim=0)
        return self.conv(x.unsqueeze(0)).view(b,c,h,w)

class CommBlock(nn.Module):
    def __init__(self, n_comm_channels, comm_dim, winsize=3, comm_type='mean') -> None:
        assert comm_type in ['mean', 'sharpen', 'maxpool']
        assert winsize % 2 == 1 and winsize>=1 , "winsize must be odd"
        super().__init__()
        self.n_channels = n_comm_channels
        self.comm_dim = comm_dim

        self.CommConv = CommConv(win_size=winsize, n_dim=self.comm_dim, comm_type=comm_type)

    def forward(self, x):
        x[:,:self.n_channels] = self.CommConv(x[:,:self.n_channels])
        return x

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        # self.weight = nn.Parameter( torch.randn(out_dim, in_dim) )
        weight = torch.empty(out_dim, in_dim) 
        torch.nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='leaky_relu')
        self.weight = nn.Parameter(weight / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        bias = self.bias*self.lr_mul if hasattr(self, 'bias') else None
        return F.linear(input, self.weight * self.lr_mul, bias=bias)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, num_comm_channels=0, comm_type='mean'):
        super().__init__()
        # TODO
        assert filters > num_comm_channels, "Number of communication channels must be less than number of filters"
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        if num_comm_channels > 0:
            # here inplace=False is important, otherwise the gradient will not be propagated
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, filters, 3, padding=1),
                leaky_relu(inplace=False),
                CommBlock(num_comm_channels, 3, comm_type=comm_type),
                nn.Conv2d(filters, filters, 3, padding=1),
                leaky_relu(inplace=False),
                CommBlock(num_comm_channels, 3, comm_type=comm_type),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, filters, 3, padding=1),
                leaky_relu(),
                nn.Conv2d(filters, filters, 3, padding=1),
                leaky_relu(),
            )
            
        # if stddev_minibatch_size > 1:
        #     self.net = nn.Sequential(
        #         MiniBatchStdDev(stddev_minibatch_size),
        #         *self.net.children()
        #     )
            
        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512,
                  comm_type='mean', comm_capacity=0, num_packs=1, minibatch_size=1, minibatch_type='original', mbstd_num_channels=0):
        super().__init__()
        assert not (minibatch_type == 'stddev' and minibatch_size == 1 and mbstd_num_channels>0), 'it is meaningless to do minibatching when minibatch_size == 1 '
        assert minibatch_type in ['original', 'stddev'], "minibatch_type must be either 'original' or 'stddev'"
        print("comm_capacity", comm_capacity)
        print("num_packs", num_packs)
        print("minibatch_size", minibatch_size)
        self.num_packs = num_packs 
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]
        comm_channels = [comm_capacity * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out_comm = list( zip(filters[:-1], filters[1:], comm_channels) )

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan, comm_chan) in enumerate(chan_in_out_comm):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out_comm) - 1)
            
            # if not is_not_last and minibatch_type == 'stddev' and minibatch_size > 1:
            #     stddev_minibatch_size = minibatch_size
            # else:
            #     stddev_minibatch_size = 0
                
            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last, 
                                       num_comm_channels=comm_chan, comm_type=comm_type)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)
        
        self.head_block = self.blocks.pop(0)
        self.head_attn = self.attn_blocks.pop(0)
        self.head_quantize = self.quantize_blocks.pop(0)

        if num_packs > 1:
            self.conn = nn.Conv2d(filters[1]*num_packs, filters[1], 3, padding=1)
        else:
            self.conn = nn.Identity()
            
        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last
        
        if minibatch_type == 'stddev' and mbstd_num_channels >= 1:
            self.final_conv = nn.Sequential(
                MinibatchStd(group_size=minibatch_size, num_channels=mbstd_num_channels),
                nn.Conv2d(chan_last+mbstd_num_channels, chan_last, 3, padding=1)
            )
        else:
            self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)

        self.flatten = Flatten()
        # minibatch block
        if minibatch_size == 1 or minibatch_type == 'stddev':
            self.to_logit = nn.Linear(latent_dim, 1)
        elif minibatch_type == 'original':
            minibatchLayer = MinibatchBlock(latent_dim, num_kernels=100, minibatch_size=minibatch_size)
            latent_dim = minibatchLayer.get_out_features()
            self.to_logit = nn.Sequential(
                minibatchLayer,
                nn.Linear(latent_dim, 1),
            )
        else:
            raise NotImplementedError
            
        # self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        x = self.head_block(x)
        if exists(self.head_attn):
            x = self.head_attn(x)
        
        if exists(self.head_quantize):
            x, loss = self.head_quantize(x)
            quantize_loss += loss

        x = self.conn(x.reshape(b//self.num_packs, -1, *x.shape[2:]))

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

# disable num_comm_channels
class StyleGAN2(nn.Module):
    def __init__(self, 
        image_size, latent_dim = 512, 
        fmap_max = 512, style_depth = 8, 
        network_capacity = 16, transparent = False, fp16 = False, 
        comm_type='mean', comm_capacity=0, num_packs=1, 
        minibatch_size=1, minibatch_type = 'original', mbstd_num_channels = 0,
        cl_reg = False, 
        steps = 1, 
        lr = 1e-4, ttur_mult = 2, betas = (0.5, 0.9), # for optimizers
        fq_layers = [], 
        fq_dict_size = 256, 
        attn_layers = [], 
        no_const = False, 
        lr_mlp = 0.1, 
        device = 0
        ):
        super().__init__()
        assert not cl_reg, "contrastive learning disabled for this project"
        # assert optimizer in ['adam', 'sgd', 'RMSprop'], "optimizer must be one of adam, sgd, RMSprop"
        assert isinstance(device, int) and device >= 0, "device must be a non-negative integer"

        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max, 
                            comm_type=comm_type, comm_capacity=comm_capacity, num_packs=num_packs, 
                            minibatch_size=minibatch_size, minibatch_type=minibatch_type, mbstd_num_channels=mbstd_num_channels)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        # if optimizer == 'adam':
        #     self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        #     self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        # elif optimizer == 'sgd':
        #     self.G_opt = SGD(generator_params, lr = self.lr, momentum=0.9)
        #     self.D_opt = SGD(self.D.parameters(), lr = self.lr * ttur_mult, momentum=0.9)
        # elif optimizer == 'RMSprop':
        #     self.G_opt = RMSprop(generator_params, lr = self.lr)
        #     self.D_opt = RMSprop(self.D.parameters(), lr = self.lr * ttur_mult)
        # else:
        #     raise ValueError('optimizer must be adam, sgd, or RMSprop')
        self.G_opt = Adam(generator_params, lr = self.lr, betas=betas)
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=betas)

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(device)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(
        self,
        name = 'default',
        data_name = 'img_align_celeba',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        comm_type = 'mean',         # type of communication.            For discriminator only.
        comm_capacity=0,            # number of communication channels. For discriminator only.
        num_packs=1,                # number of packs.                  For discriminator only.
        minibatch_size = 4,         # minibase size.                    For discriminator only.
        minibatch_type = 'original', # 'original' or 'stddev'.          For discriminator only.
        mbstd_num_channels = 0,     # for minibatchstd only.
        batch_size = 4,
        mixed_prob = 0.9,
        n_critic = 1,
        gradient_accumulate_every=1,
        lr = 2e-4, lr_mlp = 0.1, ttur_mult = 2,     # for optimizers
        G_reg_interval = 4, D_reg_interval = 16,    # for regularization
        pl_weight = 2, gp_gamma = 10, r1_gamma = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        loss_type = 'hinge',
        # dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False, device = 0, is_main = True, world_size = 1, # distributed training
        log = False,
        audacious=False,            # this is for some audacious attempts, will be deprecated in final version.
        *args,
        **kwargs
    ):
        assert 32 % n_critic == 0, 'This sounds unreasonable. But this is to make sure gradient penalty is applied every 32 iterations.'
        assert batch_size % num_packs == 0, 'batch size on each gpu must be divisible by num_packs'
        assert batch_size % minibatch_size == 0, 'batch size on each gpu must be divisible by minibatch_size'
        assert minibatch_type in ['original', 'stddev'], "minibatch type must be 'original' or 'stddev'"
        assert loss_type in ['hinge', 'bce', 'dual_contrast', 'wasserstein'], 'loss_type must be one of [hinge, bce, dual_contrast, wasserstein]'
        assert aug_prob >= 0 and aug_prob <= 1, 'aug_prob must be between 0 and 1'
        assert dataset_aug_prob >= 0 and dataset_aug_prob <= 1, 'dataset_aug_prob must be between 0 and 1'
        assert isinstance(device, int) and device >= 0, 'device must be a non-negative integer'
        
        self.audacious = audacious

        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.data_name = data_name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent
        
        self.comm_type = comm_type
        self.comm_capacity = comm_capacity
        self.num_packs = num_packs
        self.minibatch_size = minibatch_size
        self.minibatch_type = minibatch_type
        self.mbstd_num_channels = mbstd_num_channels

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types
        self.aug_kwargs = {'prob': self.aug_prob, 'types': self.aug_types}

        # self.optimizer = optimizer
        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob
        
        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval
        
        self.pl_weight = pl_weight
        self.gp_gamma  = gp_gamma
        self.r1_gamma  = r1_gamma

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.n_critic = n_critic
        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        # self.dual_contrast_loss = dual_contrast_loss
        self.loss_type = loss_type
        
        # setup losses
        if self.loss_type == 'hinge':
            self.D_loss_fn = hinge_loss
            self.G_loss_fn = gen_hinge_loss
            self.G_requires_reals = False
        elif self.loss_type == 'dual_contrast':
            self.D_loss_fn = dual_contrastive_loss
            self.G_loss_fn = dual_contrastive_loss
            self.G_requires_reals = True
        elif self.loss_type == 'bce':
            self.D_loss_fn = bce_loss
            self.G_loss_fn = gen_bce_loss
            # raise NotImplementedError
            self.G_requires_reals = False
        elif self.loss_type == 'wasserstein':
            self.D_loss_fn = w_loss
            self.G_loss_fn = gen_w_loss
            self.G_requires_reals = False
            
        self.backwards = partial(loss_backwards, self.fp16)

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = is_main
        self.device = device
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if (log and is_main) else None

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 
                'comm_type':self.comm_type, 'comm_capacity':self.comm_capacity, 'num_packs':self.num_packs, 
                'minibatch_size':self.minibatch_size, 'minibatch_type':self.minibatch_type, 'mbstd_num_channels':self.mbstd_num_channels}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # optimizer = 'RMSprop' if self.loss_type == 'wasserstein' else 'adam'
        betas = (0, 0.9) if self.loss_type == 'wasserstein' else (0.5, 0.9)
        self.GAN = StyleGAN2(
            lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, betas=betas,
            image_size = self.image_size, network_capacity = self.network_capacity, 
            fmap_max = self.fmap_max, transparent = self.transparent, 
            comm_type=self.comm_type, comm_capacity=self.comm_capacity, num_packs= self.num_packs, 
            minibatch_size=self.minibatch_size, minibatch_type=self.minibatch_type, mbstd_num_channels=self.mbstd_num_channels,
            fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, 
            fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, device = self.device, 
            *args, **kwargs)

        # if self.is_ddp:
        #     ddp_kwargs = {'device_ids': [self.device]}
        #     self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
        #     self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
        #     self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
        #     self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)
        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.device]}
            self.S = DDP(self.GAN.S, **ddp_kwargs)
            self.G = DDP(self.GAN.G, **ddp_kwargs)
            self.D = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug = DDP(self.GAN.D_aug, **ddp_kwargs)
        else:
            self.S = self.GAN.S
            self.G = self.GAN.G
            self.D = self.GAN.D
            self.D_aug = self.GAN.D_aug


        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']

        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        self.comm_type = config.pop('comm_type', self.comm_type)
        self.comm_capacity = config.pop('comm_capacity', self.comm_capacity)
        self.num_packs = config.pop('num_packs', self.num_packs)
        self.minibatch_size = config.pop('minibatch_size', self.minibatch_size)
        self.minibatch_type = config.pop('minibatch_type', self.minibatch_type)
        self.mbstd_num_channels = config.pop('mbstd_num_channels', self.mbstd_num_channels)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 
                'lr_mlp': self.lr_mlp, 'transparent': self.transparent,
                'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const,
                'comm_type':self.comm_type, 'comm_capacity':self.comm_capacity, 'num_packs': self.num_packs, 
                'minibatch_size': self.minibatch_size, 'minibatch_type': self.minibatch_type, 'mbstd_num_channels':self.mbstd_num_channels
                }

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        # num_workers = num_workers = default(self.num_workers, NUM_CORES)
        # sampler = DistributedSampler(self.dataset, rank=self.device, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        sampler = DistributedSampler(self.dataset, shuffle=True) if self.is_ddp else None # we don't need to specify rank and world_size, it will be done automatically because we have set it in dist
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        # dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = self.batch_size, shuffle = True, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        # total_disc_loss = torch.tensor(0.).cuda(self.device)
        # total_gen_loss = torch.tensor(0.).cuda(self.device)

        batch_size = math.ceil(self.batch_size / self.world_size)
        # batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        # aug_prob   = self.aug_prob
        # aug_types  = self.aug_types
        # aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        # apply_gradient_penalty  = (self.steps % self.D_reg_interval == 0) if self.loss_type != 'wasserstein' else True
        apply_gradient_penalty  = self.loss_type == 'wasserstein'
        apply_path_penalty      = not self.no_pl_reg and self.steps > 5000 and self.steps % self.G_reg_interval == 0
        apply_r1_reg            = self.steps % self.D_reg_interval == 0 if self.loss_type != 'wasserstein' else False
        apply_cl_reg_to_generated = self.steps > 20000

        # S = self.GAN.S if not self.is_ddp else self.S_ddp
        # G = self.GAN.G if not self.is_ddp else self.G_ddp
        # D = self.GAN.D if not self.is_ddp else self.D_ddp
        # D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        # S = self.GAN.S
        # G = self.GAN.G
        # D = self.GAN.D
        # D_aug = self.GAN.D_aug

        # backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
                    noise = image_noise(batch_size, image_size, device=self.device)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda(self.device)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            self.backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # avg_pl_length = self.pl_mean
        
        # train discriminator
        with no_grad(self.S, self.G):
            total_disc_loss = self._train_discriminator(batch_size, apply_gradient_penalty, apply_r1_reg)

        # train generator
        if self.steps % self.n_critic == 0:
            with no_grad(self.D_aug):
                total_gen_loss, avg_pl_length = self._train_generator(batch_size, apply_path_penalty=apply_path_penalty)
        else:
            total_gen_loss = 0

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                # fid = self.calculate_fid(self.calculate_fid_num_images)
                
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None
    
    def _train_discriminator(self, batch_size, apply_gradient_penalty, apply_r1_reg):
        total_disc_loss = torch.tensor(0.).cuda(self.device)
        self.GAN.D_opt.zero_grad()
        
        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[self.D_aug, self.S, self.G]):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            w_space = latent_to_w(self.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.G(w_styles, noise)
            fake_output, fake_q_loss = self.D_aug(generated_images.clone().detach(), detach = True, **self.aug_kwargs)

            image_batch = next(self.loader).cuda(self.device)
            image_batch.requires_grad_()
            real_output, real_q_loss = self.D_aug(image_batch, **self.aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = self.D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output, weight=self.gp_gamma)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp
                
            if apply_r1_reg:
                r1_loss = r1_reg(image_batch, real_output, gamma=self.r1_gamma)
                self.last_r1_loss = r1_loss.clone().detach().item()
                self.track(self.last_r1_loss, 'R1reg')
                disc_loss = disc_loss + r1_loss

        disc_loss = disc_loss / self.gradient_accumulate_every
        disc_loss.register_hook(raise_if_nan)
        self.backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

        total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.GAN.D_opt.step()
        
        return total_disc_loss
       
    def _train_generator(self, batch_size, apply_path_penalty):
        total_gen_loss = torch.tensor(0.).cuda(self.device)
        
        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers
        
        self.GAN.G_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[self.S, self.G, self.D_aug]):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            
            w_space = latent_to_w(self.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.G(w_styles, noise)
            fake_output, _ = self.D_aug(generated_images, **self.aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if self.G_requires_reals:
                image_batch = next(self.loader).cuda(self.device)
                real_output, _ = self.D_aug(image_batch, detach = True, **self.aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = self.G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean() * self.pl_weight
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss
            else:
                avg_pl_length = self.pl_mean
                
            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            self.backwards(gen_loss, self.GAN.G_opt, loss_id = 2)
            del gen_loss

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        self.GAN.G_opt.step()
        
        return total_gen_loss, avg_pl_length

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.device)
        n = image_noise(num_rows ** 2, image_size, device=self.device)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.device)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.device)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def get_model_features(self, mode="clean", num_gen=50_000):
        from cleanfid.resize import build_resizer
        from cleanfid.features import build_feature_extractor
        fn_resize = build_resizer(mode)
        feat_model = build_feature_extractor(mode, device=self.device, use_dataparallel=not self.is_ddp)
        
        def _resize_batch(x):
            resized_batch = []
            for i in range(x.shape[0]):
                img = x[i].cpu().numpy().transpose((1, 2, 0))
                img = fn_resize(img)
                resized_batch.append( torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0) )
            return torch.cat(resized_batch, dim=0).to(self.device)
        
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers
        
        # generate test features
        num_iters = int(np.ceil(num_gen / self.batch_size))
        l_feats = []
        
        for idx in range(num_iters):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(self.batch_size, image_size, device=self.device)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)
            if mode != "legacy_tensorflow":
                resized_images = _resize_batch(generated_images)
            else:
                resized_images = generated_images
            
            # get_batch_features
            feat = feat_model(resized_images).detach().cpu().numpy()
            l_feats.append(feat)
            
        return np.concatenate(l_feats)[:num_gen]
    
    # @torch.no_grad()
    # def calculate_fid(self, num_gen=50_000, mode="clean"):
    #     # from cleanfid.features import get_reference_statistics
    #     print("Calculating FID...")
    #     feat = self.get_model_features(mode=mode, num_gen=num_gen//self.world_size)
        
    #     if self.is_ddp:
    #         rmtree(self.fid_dir, ignore_errors=True)
    #         os.mkdir(self.fid_dir)
            
    #         np.save( str(self.fid_dir / f"feat_{self.device}.npy"), feat)
    #         print(f"Saved features to {self.fid_dir} on device {self.device}")  # for debugging
    #         torch.distributed.barrier()     # wait for all processes to save their features
            
    #         for path in self.fid_dir.glob("feat_*.npy"):
    #             if path.name == f"feat_{self.device}.npy":
    #                 continue
    #             feat = np.concatenate([feat, np.load(path)])
                
            
    #     if self.is_main:    
    #         ref_mu, ref_sigma = cleanfid.features.get_reference_statistics(res=self.image_size, name=self.data_name, mode=mode, split="custom")
    #         mu,     sigma     = np.mean(feat, axis=0), np.cov(feat, rowvar=False)
    #         return cleanfid.fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)
    #     else:
    #         return None
                
    
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        assert self.is_main
        # from pytorch_fid import fid_score
        from cleanfid import fid
        torch.cuda.empty_cache()

        # real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers
        
        print('calculating FID...')

        # for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
        for batch_num in range(num_batches):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(self.batch_size, image_size, device=self.device)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        # return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)
        return fid.compute_fid(str(fake_path), dataset_name = self.data_name, mode = 'clean' , dataset_split="custom", \
                               use_dataparallel = not self.is_ddp, verbose = False, device = self.device, num_workers = 0)

    @torch.no_grad()
    def truncate_style(self, tensor, S, trunc_psi = 0.75):
        # S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.device)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.device)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, S, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, S = S, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, S = S, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.device)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.device)
        n = image_noise(num_rows ** 2, image_size, device=self.device)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('R1reg', self.last_r1_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        if not self.audacious:
            self.load_config()
            strict = True
        else:
            strict = False

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'], strict=strict)
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'], strict=strict)

class ModelLoader:
    def __init__(self, *, base_dir, name = 'default', load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images

if __name__ == '__main__':
    # test unit for minibatch block
    
    # mb = MinibatchBlock(in_features=20, num_kernels=5, dim_per_kernel=3, minibatch_size=8)
    # x = torch.randn(32, 20)
    # out = mb(x)
    # print(mb.get_out_features())
    
    mb = MinibatchStd(group_size=4)
    x  = torch.randn(32, 124, 4, 4)
    out = mb(x)
    print(out.shape)
    
from utils import *
from matplotlib import pyplot as plt
from stylegan2_pytorch import Trainer, StyleGAN2
from argparse import Namespace, ArgumentParser
from pytorch_fid import fid_score
from version import __version__
import shutil
import os
import glob

from torch.utils.tensorboard import SummaryWriter

class Analyzer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'reports',
        models_dir = 'models',
        base_dir = './',
        data_dir = './data',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        comm_type='mean',       # communication type.               For discriminator only.   
        comm_capacity=0,        # number of communication channels. For discriminator only.
        num_packs=1,            # number of packs.                  For discriminator only.
        minibatch_size=1,       # minibatch size.                   For discriminator only.
        batch_size = 4,
        gradient_accumulate_every=1,
        num_workers = None,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        loss_type = 'hinge',
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        rank = 0,
        log = False,
        *args,
        **kwargs
    ):
        assert isinstance(rank, int) and rank >= 0, 'rank must be a non-negative integer'

        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

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


        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_image_tiles = num_image_tiles
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.pl_length_ma = EMA(0.99)
        # self.init_folders()

        # self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.loss_type = loss_type

        self.rank = rank

        self.logger = SummaryWriter(log_dir=self.results_dir / name) if log else None

        self.load_config()
        self.set_data_src(data_dir)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return len(list(self.models_dir.glob(f'{self.name}/*.pt')))

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 
                'comm_type':self.comm_type, 'comm_capacity':self.comm_capacity, 'num_packs':self.num_packs, 'minibatch_size':self.minibatch_size}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # here optimizer, lr, lr_mlp, ttur_mult no longer matters.
        self.GAN = StyleGAN2(
            optimizer='adam', lr = 1, lr_mlp = self.lr_mlp, ttur_mult = 1, 
            image_size = self.image_size, network_capacity = self.network_capacity, 
            fmap_max = self.fmap_max, transparent = self.transparent, comm_capacity=self.comm_capacity, num_packs= self.num_packs, minibatch_size=self.minibatch_size,
            fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, 
            fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, 
            *args, **kwargs)

        # if self.is_ddp:
        #     ddp_kwargs = {'device_ids': [self.rank]}
        #     self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
        #     self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
        #     self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
        #     self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        # if exists(self.logger):
        #     self.logger.set_params(self.hparams)

    def load_config(self):
        config = json.loads(self.config_path.read_text())
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
        self.comm_papacity = config.pop('comm_capacity', self.comm_capacity)
        self.num_packs = config.pop('num_packs', self.num_packs)
        self.minibatch_size = config.pop('minibatch_size', self.minibatch_size)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 
                'lr_mlp': self.lr_mlp, 'transparent': self.transparent,
                'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const,
                'comm_type':self.comm_type, 'comm_capacity':self.comm_capacity, 'num_packs': self.num_packs, 'minibatch_size': self.minibatch_size
                }
        
    def analyse_fid(self, num=-1):
        self.load(num)
        num_repetitions = 50
        
        fids = []
        intra_fids = []
        for i in range(num_repetitions):
            fid = self.calculate_fid(num_batches = self.calculate_fid_num_images // self.batch_size)
            intra_fid = self.calculate_intra_fid()
            fids.append(fid)
            intra_fids.append(intra_fid)
            
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        with open(self.results_dir / f'{self.name}_fid.txt', 'w') as f:
            f.write(f'{np.mean(fids)}, {np.std(fids)}, {np.mean(intra_fids)}, {np.std(intra_fids)}\n')
            
        return np.mean(fids), np.std(fids), np.mean(intra_fids), np.std(intra_fids)
    
    # def analyse_fids(self, final=True):
    #     if final:
    #         self.load()
    #         fid = self.calculate_fid(num_batches = self.calculate_fid_num_images // self.batch_size)
    #         intra_fid = self.calculate_intra_fid()
    #         return fid, intra_fid
        
    #     fids = []
    #     intra_fids = []
    #     for i in range(self.checkpoint_num):
    #         self.load(i)
    #         fid = self.calculate_fid(num_batches = self.calculate_fid_num_images // self.batch_size)
    #         intra_fid = self.calculate_intra_fid()
    #         fids.append(fid)
    #         intra_fids.append(intra_fid)
    #         if exists(self.logger):
    #             self.logger.add_scalar('fid', fid, i)
    #             self.logger.add_scalar('intra_fid', intra_fid, i)

    #         plt.figure(figsize=(10, 5))
    #         plt.plot(fids, label='fid')
    #         plt.plot(intra_fids, label='intra_fid')
    #         plt.legend()
    #         plt.savefig(self.results_dir / self.name / 'fid.png')

    #     return fids, intra_fids

    def set_data_src(self, folder):
        # we should not augment the dataset when doing evaluation
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = 0)
        num_workers = num_workers = default(self.num_workers, NUM_CORES)
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = self.batch_size, shuffle = True, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # # auto set augmentation prob for user if dataset is detected to be low
        # num_samples = len(self.dataset)
        # if not exists(self.aug_prob) and num_samples < 1e5:
        #     self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
        #     print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

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
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension



        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # style = noise_list(batch_size, num_layers, latent_dim, device=self.rank)
        # noise = image_noise(batch_size, image_size, device=self.rank)

        # w_space = latent_to_w(self.GAN.S, style)
        # w_styles = styles_def_to_tensor(w_space)

        # generated_images = self.GAN.G(w_styles, noise)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.GE(w_styles, noise)
            # generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            # torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}.{ext}'))

        return fid_score.calculate_fid_given_paths(paths=[str(real_path), str(fake_path)], batch_size=64, device=noise.device, dims=2048)

    @torch.no_grad()
    def calculate_intra_fid(self):
        # this function is called fater calculate_fid, so we can assume that the fake images are already generated
        fake_path = self.fid_dir / 'fake'
        # list the num of fake images using glob
        num_img   = len(glob.glob(str(fake_path / '*')))

        # split the fake images into two sets
        fake_path1 = self.fid_dir / 'fake1'
        fake_path2 = self.fid_dir / 'fake2'

        # fid_scores = []

        rmtree(fake_path1, ignore_errors=True)
        rmtree(fake_path2, ignore_errors=True)

        # rename fake_path to fake_path1
        # os.rename(fake_path, fake_path1)
        os.mkdir(fake_path1)
        os.mkdir(fake_path2)

        intra_fids = []
        for i in range(5):
            fake1_nums = np.random.choice(num_img, num_img // 2, replace=False)

            for i in range(num_img):
                if i in fake1_nums:
                    shutil.copy(str(fake_path / f'{str(i)}.{self.image_extension}'), str(fake_path1 / f'{str(i)}.{self.image_extension}'))
                else:
                    shutil.copy(str(fake_path / f'{str(i)}.{self.image_extension}'), str(fake_path2 / f'{str(i)}.{self.image_extension}'))
                    
            intra_fids.append(fid_score.calculate_fid_given_paths(paths=[str(fake_path1), str(fake_path2)], batch_size=64, device=self.rank, dims=2048))

        # calculate the fid score between fake_path1 and fake_path2
        return np.mean(intra_fids)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
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

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

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

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            # print(f'continuing from previous epoch - {name}')

        load_data = torch.load(self.model_name(name))
        print(f'loading from save point {name} ...')

        # if 'version' in load_data:
        #     print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'], strict=False)
            # we don't need discriminator for analysis
            del self.GAN.D
            del self.GAN.D_aug
            del self.GAN.D_cl
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])

        self.GAN.eval()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/home/zichuan/data/img_align_celeba')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--calculate_fid_num_images', type=int, default=6400)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=None)
    # parser.add_argument('--trunc_psi', type=float, default=0)
    parser.add_argument('--rank', type=int, default=1)

    arg = parser.parse_args()
    
    # names = ['b32_hinge_cc1', 'b32_bce_pac2', 'b32_bce_pac4', 'b32_dcl_pac2', 'b32_dcl_pac4']
    names = ['b32_bce_cc1']
    for name in names:
        analyzer = Analyzer(
            name=name,
            data_dir=arg.data_dir,
            batch_size=arg.batch_size,
            calculate_fid_num_images = arg.calculate_fid_num_images,
            log = arg.log,
            num_workers = arg.num_workers,
            # trunc_psi=arg.trunc_psi,
            rank=arg.rank,
            clear_fid_cache=True
        )
        # to test if the model can be loaded correctly
        analyzer.load() 
    
    
    with open('/home/zichuan/stylegan2-pytorch/results.csv', 'a') as f:
        f.write('names, fid, fid_std, intra_fid, intra_fid_std \n')

    for name in names:
        analyzer = Analyzer(
            name=name,
            data_dir=arg.data_dir,
            batch_size=arg.batch_size,
            calculate_fid_num_images = arg.calculate_fid_num_images,
            log = arg.log,
            num_workers = arg.num_workers,
            # trunc_psi=arg.trunc_psi,
            rank=arg.rank,
            clear_fid_cache=True
        )
        
        fid, fid_std, intra_fid, intra_fid_std = analyzer.analyse_fid()
        
        with open('/home/zichuan/stylegan2-pytorch/results.csv', 'a') as f:
            f.write(f'{name}, {fid}, {fid_std}, {intra_fid}, {intra_fid_std} \n')

     
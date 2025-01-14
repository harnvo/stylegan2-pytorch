import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylegan2_pytorch import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, devices, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    world_size = len(devices)
    is_ddp = world_size > 1
    # print('rank', rank, 'is_main', is_main)

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        device = devices[rank],
        is_main = is_main,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()
        
    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>') if is_main else None
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        # if model.steps % 10 == 0 and progress_bar is not None:
        if progress_bar is not None:
            progress_bar.n = model.steps
            progress_bar.refresh()
        if model.steps % 50 == 0 and is_main:
            model.print_log()  
        
    print("saving...")
    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 128,
    network_capacity = 16,
    fmap_max = 512,
    transparent = False,
    comm_type = 'mean', comm_capacity = 0, num_packs = 1, minibatch_size=1, mbstd_num_channels=0, minibatch_type='stddev', # for discriminator only
    batch_size = 32,
    n_critic = 1,
    gradient_accumulate_every = 1,
    num_train_steps = 5_000_000,
    learning_rate = 2e-4, lr_mlp = 0.1, ttur_mult = 1.5, # lr_mlp: Learning rate multiplier for the mapping layers.
    G_reg_interval = 4, D_reg_interval = 16,    # for regularization
    pl_weight = 2, gp_gamma = 2, r1_gamma = 2,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 5000, evaluate_every = 5000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    fp16 = False,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    # aug_types = ['translation', 'cutout'],
    aug_types = [],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    loss_type = 'hinge',
    dataset_aug_prob = 0.,
    devices = [0],
    calculate_fid_every = None,
    calculate_fid_num_images = 50_000,
    clear_fid_cache = False,
    seed = 0,
    log = False,
    audacious = False
):
    if name == 'default':
        name = f"b{batch_size}_{loss_type}"
        if comm_capacity > 0:
            name += f"_cc{comm_capacity}"
            name += f"{comm_type}"
        if num_packs > 1:
            name += f"_p{num_packs}"
        if minibatch_size > 1:
            name += f"_mb{minibatch_size}"
            name += 'stddev' if minibatch_type == 'stddev' else ''
        if ttur_mult != 1.5:
            name += f"_t{ttur_mult}"
            
    if devices == 'all':
        num_devices = torch.cuda.device_count()
        devices = list(range(num_devices))
            
    data_name = str(os.path.basename(data))        
    from cleanfid import fid
    if calculate_fid_every is not None and not fid.test_stats_exists(data_name, mode='clean'):
        fid.make_custom_stats(name=data_name, fdir = data, mode='clean')
        
    model_args = dict(
        name = name,
        data_name = data_name,          # for computing FID
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        n_critic = n_critic,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        comm_type = comm_type, comm_capacity = comm_capacity, num_packs = num_packs, 
        minibatch_size = minibatch_size, mbstd_num_channels = mbstd_num_channels ,minibatch_type = minibatch_type,
        lr = learning_rate, lr_mlp = lr_mlp, ttur_mult = ttur_mult,
        G_reg_interval = G_reg_interval, D_reg_interval = D_reg_interval,
        pl_weight = pl_weight, gp_gamma = gp_gamma, r1_gamma = r1_gamma,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        loss_type = loss_type,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log,
        audacious=audacious
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return
    
    # run_training(device, model_args, data, load_from, new, num_train_steps, name, seed)

    assert 0 < len(devices) <= torch.cuda.device_count(), f'invalid device list {devices}'
    world_size = len(devices)

    if world_size == 1:
        run_training(0, devices, model_args, data, load_from, new, num_train_steps, name, seed)
        return

    mp.spawn(run_training,
        args=(devices, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)

def main():
    fire.Fire(train_from_folder)

if __name__ == '__main__':
    main()

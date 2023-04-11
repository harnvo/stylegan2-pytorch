python stylegan2_pytorch/cli.py --gradient_accumulate_every 1 --data ../data/img_align_celeba --num_workers 32 \
    --num_train_steps 1000000 --save_every 20000 --evaluate_every 10000 --calculate_fid_every 10000 --calculate_fid_num_images 50000 \
    --devices 'all' --aug_prob 0.5 \
    --learning_rate 2e-4 --lr_mlp 0.1 --comm_type mean --comm_capacity 1 --mbstd_num_channels 1 --new
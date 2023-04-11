python stylegan2_pytorch/cli.py --gradient_accumulate_every 1 --data ../data/img_align_celeba --num_workers 32\
    --num_train_steps 1000000 --save_every 20000 --evaluate_every 20000 --calculate_fid_every 10000 --calculate_fid_num_images 128\
    --learning_rate 2e-4 --ttur_mult 1 --lr_mlp 0.01  --loss_type hinge\
    --devices [0] --aug_prob 0.5 --name debug --image_size 64 --new
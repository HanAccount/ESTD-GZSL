python train.py --dataset CUB  --ga 1 --beta 0.003 --dis 0.3 --nSample 1200 --gpu 0 --S_dim 2048 --NS_dim 2048 --lr 0.00003  --classifier_lr 0.002 --gen_nepoch 600 --kl_warmup 0.001 --tc_warmup 0.0001 --weight_decay 1e-8 --vae_dec_drop 0.1 --dis_step 3 --ae_drop 0.0 --finetune true --class_embedding sent --recon_weight 0.1

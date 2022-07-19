export CUDA_VISIBLE_DEVICES=6,7

python run_nerf_uncertainty_NF.py \
            --config configs/basket_ds.txt \
            --expname 'basket' \
            --N_rand 512 \
            --N_samples 128 \
            --n_flows 4 \
            --h_alpha_size 64 \
            --h_rgb_size 64 \
            --K_samples 32 \
            --n_hidden 128 \
            --type_flows 'triangular' \
            --beta1 0.01 \
            --depth_lambda 0.01 \
            --netdepth 8 \
            --netwidth 512 \
            --model 'NeRF_Flows' \
            --index_step -1 \
            # --is_train \
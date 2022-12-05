#!/user/bin/env bash

set -e
set -x
export NCCL_LL_THRESHOLD=0


rank_num=$1
rank_index=$2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=$rank_num \
	--node_rank=$rank_index --master_addr="100.109.33.223" --master_port=3349 \
	with_tricks_train.py  ../imagenet_raw/ --model resnet50 -b 256 --sched cosine --epochs 300 \
	--reinforce 0.3 --kl_start_epoch 100 --kl_end_epoch 300 --CE_reinforce 2 --CE_start_epoch 20 --KL_all_indices\
	--opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 10  \
	--model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel \
	--reprob 0.3 --lr 40e-4 --min-lr 1e-6 --weight-decay 0.05 --drop 0.0 \
	--drop-path .35 --img-size 224 --mixup 0.8 --cutmix 1.0 \
	--smoothing 0.1 \
	--output ../SLL_output/NO_inear_decay_intersect_90E \
	--EMA_Vanilla_intersection \
	--amp --model-ema \
	# --resume ../fan_base_16_p4_convnext/train/20220219-074922-fan_base_16_p4_convnext-224/model_best.pth.tar \
	# --resume ../fan_small_18_p16_c448_continued/train/20220125-081518-fan_small_18_p16_224-224/checkpoint-317.pth.tar \
	# --start-epoch 200 \
	# --resume ../fan_small_24_p16/train/20220122-064940-fan_small_18_p16_224-224/model_best.pth.tar \
	# --resume ../fan_small_12_p16_224_sr_2_median_pool/train/20211117-145637-fan_small_12_p16_224-224/model_best.pth.tar 
	# --resume ../fan_small_12_p8_224_no_conv_blk_one_projection_for_all_blocks_sr_1/train/20211030-163521-fan_small_12_p8_224-224/model_best.pth.tar
	# --resume ../fan_small_12_p16_224_no_conv_blk_no_projection_sr_2/train/20211022-070808-fan_small_12_p16_224-224/model_best.pth.tar \
	# --resume ../robust_pvt_v2_b2_xcit_attn_spatial_channel_fix_sampling_no_ca_v1/train/20211006-151250-robust_pvt_v2_b2_emlp_v1_spatial_channel-224/model_best.pth.tar \
	# --cos_reg_component 0  --save_code --amp \
	# --resume ../refiner_pvt_v2_b2_emlp_2_stages_pos_embed_add_norm_move_conv_conv_patch_processing/train/20210904-132925-refiner_pvt_v2_b2_emlp_v1-224/checkpoint-187.pth.tar
	# --resume ../refiner_pvt_v2_b2_emlp_no_norm_no_cos_reg_transpose/train/20210830-082556-refiner_pvt_v2_b2_emlp_v1-224/model_best.pth.tar
	# --resume ../output/pvt_v2/pvt_v2_b2.pth
	# --output ../refiner_output/revit_16b_6h_no_pos_embed  \
	# --resume ../small_mlp_refiner_pvt_v2_2_stage_replace/train/20210822-014816-refiner_pvt_v2_b2_emlp-224/model_best.pth.tar  #--clip-grad 0.2 --clip-mode agc
	# --relabel --relabel-data ../label_top5_train_nfnet/ --relabel-size 14 --dense-weight 0.5 \
python3 idle.py

#!/bin/bash
./distributed_train.sh 8 ../imagenet_raw --model deit_small_patch16_224 --epochs 300 --sched cosine \
	--reinforce 0.3 --kl_start_epoch 60 --kl_end_epoch 300 --CE_reinforce 2 --CE_start_epoch 300 \
	--model-ema --smoothing 0.1 --lr 20e-4 -b 256 \
	--output ../SLL_output/NO_inear_decay_intersect_300E_tricks \
        --model-ema-decay 0.9992 --weight-decay 5e-2 \
	-j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
	--aa rand-m9-mstd0.5-inc1 --remode pixel \
	--reprob 0.25 --min-lr 1e-6 \
	--drop-path .0 --img-size 224 --mixup 0.2 --cutmix 1.0 \
	--smoothing 0.1 \
        --opt adamw  \
	--minus_epoch 3 \
	# --EMA_Vanilla_intersection \
	# --KL_all_indices \
	# --resume ../SLL_output/NO_inear_decay_intersect_300E_tricks/20220227-015232-resnet50-224/model_best.pth.tar \
	# --resume ../SLL_output/NO_inear_decay_intersect_300E_tricks/20220215-022524-resnet50-224/checkpoint-103.pth.tar \
	# --resume ../SLL_output/NO_inear_decay_intersect_300E_tricks/20220114-135110-resnet50-224/model_best.pth.tar \
	# --CE_linear_decay \
	# --KL_linear_decay \
        # --resume ../SLL_output/train/20211231-034316-resnet50-224/checkpoint-23.pth.tar \
python3 idle.py

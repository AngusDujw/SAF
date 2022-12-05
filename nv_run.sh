#!/bin/bash
./distributed_train.sh 8 ../imagenet_raw --model resnet50 --epochs 90 --sched cosine \
	--reinforce 0.3 --kl_start_epoch 10 --kl_end_epoch 100 --CE_reinforce 2 --CE_start_epoch 20 \
	--model-ema --smoothing 0.1 --lr 1.4 -b 256 \
	--output ../SLL_output/NO_inear_decay_intersect_90E --amp \
        --model-ema-decay 0.9998 --weight-decay 3e-5 \
	--EMA_Vanilla_intersection \
	--minus 3 \
        # --CE_linear_decay \
	# --KL_linear_decay \
        # --resume ../SLL_output/train/20211231-034316-resnet50-224/checkpoint-23.pth.tar \
python3 idle.py

   
 #CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 SAF.py /home/djw/data/imagenet --model resnet50 --epochs 90 -b 256 --sched cosine --reinforce 0.3 --kl_start_epoch 5  --model-ema --smoothing 0.1 --minus_epoch 2 --lr 0.4 --weight-decay 1e-4


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 MESA_train.py /home/djw/data/imagenet --model resnet50 --epochs 90 -b 256 --sched cosine --reinforce 0.8 --kl_start_epoch 5 --model-ema --smoothing 0.1  --lr 0.4 --weight-decay 1e-4



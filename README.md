# Sharpness-aware Training for Free 

Code for [“Sharpness-aware Training for Free”](https://arxiv.org/abs/2205.14083), which has been accepted by NeurIPS 2022. 



## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.8.8
- torch = 1.8.0
- torchvision = 0.9.0
- 
## File description 
The current version is embeded with [“TIMM”](https://github.com/rwightman/pytorch-image-models). The modularized version will be updated later. 

SAF_train.py , MESA_train.py are for SAF and MESA training, respectively. 

Check run.sh for two examples and parameters setting. 

Recommended to use vanilla pytorch DDP. 

The main modification is on SAF_train.py, MESA_train.py, the rest modification is making the data_loader be able to output unique index of each sample. Therefore, the past outputs can be taken as the reference to construct the trajectory loss of SAF. (vanilla batch output: img, label. Modified batch output: indice,img,label) 



## Hyperparamters
() represents the default values


--reinforce (0.3/0.8) lambda, the coefficient of the trajectory loss. 

--temperature (5) temperature, the temperature of the trajectory loss. A very stable parameter, no need to tune for the most of scenarios. 

--kl_start_epoch (5) E_start, from which epoch to apply the trajectory loss. It is recommended to be the beginning epoch (5), or the intermediate epoch (30% of the total epochs) 

--minus_epoch (2/3) \tilde{E}, the reference soft labels of SAF. 







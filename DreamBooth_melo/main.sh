#CUDA_VISIBLE_DEVICES=3 python lora_run.py +algs=lora_diff +experiment=diffusion +model=dreambooth
#CUDA_VISIBLE_DEVICES=3 python melo_run.py +algs=melo_diff +experiment=diffusion +model=dreambooth
#CUDA_VISIBLE_DEVICES=3 python ft_run.py +algs=ft_diff +experiment=diffusion +model=dreambooth
CUDA_VISIBLE_DEVICES=3 python ft_load_generate.py +algs=ft_diff +experiment=diffusion +model=dreambooth
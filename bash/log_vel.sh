#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=webdojo-eval                                                                                                                                
#SBATCH --output=/home/taywonmin/slurm-logs/test-%j.log  # log                                                                                                   
#SBATCH --error=/home/taywonmin/slurm-logs/test-%j.log   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=2     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=8G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행  

cd /home/taywonmin/rsec/LIBERO

python eval.py \
    --save_velocity \
    --save_position
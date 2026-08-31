```shell
  cd /data/home/yuanhaoc/
  wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
  bash Anaconda3-2024.02-1-Linux-x86_64.sh  # 安装路径选 /data/home/yuanhaoc/tools/anaconda3
  export PATH=/data/home/yuanhaoc/tools/anaconda3/bin:$PATH
  source /data/home/yuanhaoc/tools/anaconda3/bin/activate
  conda create -n myenv python=3.9
  conda activate myenv
```
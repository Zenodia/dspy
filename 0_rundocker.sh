sudo docker run --gpus all  -it --rm -v $(pwd):/workspace --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --net=host  --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:24.04-py3
#xrdemo:latest /bin/bash #nvcr.io/nvidia/pytorch:23.05-py3 #demo_guardrails:latest /bin/bash  #nvcr.io/nvidia/pytorch:23.05-py3

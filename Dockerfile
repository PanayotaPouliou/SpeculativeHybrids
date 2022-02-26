# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.202.5/containers/ubuntu/.devcontainer/base.Dockerfile

#FROM  pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
#FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# [Optional] Uncomment this section to install additional OS packages.
 RUN apt-get update -y && export DEBIAN_FRONTEND=noninteractive \     
     && apt-get -y install --no-install-recommends apt-utils git curl ca-certificates libglib2.0-0 cmake iotop libjpeg8-dev  \
     && apt-get -y install --no-install-recommends libsm6 libxext6 libxrender-dev libyaml-dev zsh wget tmux ffmpeg

 RUN apt-get install ffmpeg libsm6 libxext6  -y

 RUN conda install -y ipython h5py nltk joblib jupyter pandas scipy \
     # Install DLNest
     && pip install git+https://github.com/SymenYang/DLNest.git \
     # Install other libs
     && pip install tensorboard \
     && pip install sklearn \
     # Set path in config
     #&& bash BeforeTrain.sh
     && pip install numpy>=1.19.5 cython matplotlib opencv-python tqdm \
     && pip install sklearn

     #&& pip install requests ninja cython yacs>=0.1.8 numpy>=1.19.5 cython matplotlib opencv-python tqdm \
     #&& pip install protobuf sklearn boto3 scikit-image cityscapesscripts \
     #&& pip install inference-schema pillow neo4j pylint\
     #&& pip --no-cache-dir install --force-reinstall -I pyyaml 

 #RUN conda install -y -c conda-forge timm einops && conda install -y -c conda-forge pycocotools


 RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
     && apt-get install -y software-properties-common \
     && apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod \
     && apt-get update

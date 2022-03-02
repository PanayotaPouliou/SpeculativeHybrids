# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.202.5/containers/ubuntu/.devcontainer/base.Dockerfile

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update -y && export DEBIAN_FRONTEND=noninteractive 
    #\     
    #&& apt-get -y install --no-install-recommends apt-utils git curl ca-certificates libglib2.0-0 cmake iotop libjpeg8-dev  \
    #&& apt-get -y install --no-install-recommends libsm6 libxext6 libxrender-dev libyaml-dev zsh wget tmux ffmpeg

#Import CV2
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install git -y

RUN conda install -y ipython h5py nltk joblib pandas scipy \
    # Install DLNest
    && pip install git+https://github.com/SymenYang/DLNest.git \
    # Install other libs
    && pip install tensorboard 
    #&& pip install numpy>=1.19.5 cython matplotlib opencv-python tqdm \

RUN pip install sklearn

RUN pip install numpy

#RUN conda install -y -c conda-forge timm einops && conda install -y -c conda-forge pycocotools


#RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
#    && apt-get install -y software-properties-common \
#    && apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod \
#    && apt-get update

# Use the official Ubuntu 20.04 image as the base
# FROM ubuntu:20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04


# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    git \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# VirtualGL Dependencies
RUN apt-get update && apt-get install -y \
    openbox \
    libxv1 \
    libglu1-mesa \
    mesa-utils \
    libglvnd-dev \
    wget \
    xvfb \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

# CoppeliaSim Dependencies
RUN apt-get update && \
    apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV CODING_ROOT="/opt/baeisner"

WORKDIR $CODING_ROOT
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$CODING_ROOT/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.9 using pyenv
RUN pyenv install 3.9.12
RUN pyenv global 3.9.12

###########################
# OLD STUFF
###########################

# Install PyTorch with CUDA support (make sure to adjust this depending on your CUDA version)
# RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install pytorch geometric.
# RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# Install pytorch3d
# RUN pip install fvcore iopath && \
#     pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html

###########################
# END OLD STUFF
###########################

# Download CoppeliaSim
RUN mkdir $CODING_ROOT/.coppelia
WORKDIR $CODING_ROOT/.coppelia
RUN curl -L https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -o CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# modify environment variables
ENV COPPELIASIM_ROOT="$CODING_ROOT/.coppelia/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
ENV QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"


# TODO: put this above the code copying.
# Install VirtualGL
RUN wget --no-check-certificate https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb \
    && dpkg -i virtualgl_*.deb \
    && rm virtualgl_*.deb

# Configure VirtualGL
RUN /opt/VirtualGL/bin/vglserver_config +s +f -t +egl

# Setup environment variables for NVIDIA and VirtualGL
ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV NVIDIA_DRIVER_CAPABILITIES all

###########################
# Special Torch Install
###########################

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install fvcore iopath && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html


# Install CFFI
RUN pip install cffi==1.14.2 wheel

# Install PyRep
RUN pip install --no-build-isolation "pyrep @ git+https://github.com/stepjam/PyRep.git"

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code
WORKDIR $CODING_ROOT/code

# Only copy in the source code that is necessary for the dependencies to install
COPY ./taxpose $CODING_ROOT/code/taxpose
COPY ./third_party $CODING_ROOT/code/third_party
COPY ./setup.py $CODING_ROOT/code/setup.py
COPY ./pyproject.toml $CODING_ROOT/code/pyproject.toml
RUN pip install -e ".[rlbench]"
RUN pip install -e third_party/ndf_robot

# Changes to the configs and scripts will not require a rebuild
COPY ./configs $CODING_ROOT/code/configs
COPY ./scripts $CODING_ROOT/code/scripts

RUN git config --global --add safe.directory /root/code

# Make a data directory.
RUN mkdir $CODING_ROOT/data

# Make a logs directory.
RUN mkdir $CODING_ROOT/logs

# Install gdown.
RUN pip install gdown

# Copy the download script.
COPY ./download_data.sh $CODING_ROOT/code/download_data.sh


# Install:


COPY ./docker/entrypoint.sh /opt/baeisner/code/entrypoint.sh
ENTRYPOINT ["/opt/baeisner/code/entrypoint.sh"]

# RUN pip install --force-reinstall torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# RUN pip install --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install --force-reinstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# RUN pip uninstall -y pytorch3d
# RUN pip install --force-reinstall fvcore iopath && \
#     pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
#
# RUN rm -r /opt/baeisner/.coppelia
# # COPY docker/.coppelia /opt/baeisner/.coppelia
# COPY docker/.coppelia/coppeliasim.tar.xz /opt/baeisner/.coppelia/coppeliasim.tar.xz
# WORKDIR $CODING_ROOT/.coppelia
# RUN tar -xf /opt/baeisner/.coppelia/coppeliasim.tar.xz
# WORKDIR $CODING_ROOT/code

# ENV COPPELIASIM_ROOT="$CODING_ROOT/.coppelia/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
# ENV QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
# RUN chmod 777 /opt/baeisner/.coppelia/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

# RUN echo "Hello"
# RUN pip install --user --no-build-isolation "pyrep @ git+https://github.com/stepjam/PyRep.git"

# RUN curl -L https://downloads.coppeliarobotics.com/V4_6_0_rev18/CoppeliaSim_Pro_V4_6_0_rev18_Ubuntu20_04.tar.xz -o CoppeliaSim_Pro_V4_6_0_rev18_Ubuntu20_04.tar.xz && \
#     tar -xf CoppeliaSim_Pro_V4_6_0_rev18_Ubuntu20_04.tar.xz && \
#     rm CoppeliaSim_Pro_V4_6_0_rev18_Ubuntu20_04.tar.xz

# Set up the entry point
CMD ["python", "-c", "import torch; print(torch.cuda.is_available())"]

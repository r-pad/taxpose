# Use the official Ubuntu 20.04 image as the base
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev \
    git \
    libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV CODING_ROOT="/opt/baeisner"

WORKDIR $CODING_ROOT
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$CODING_ROOT/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.10 using pyenv
RUN pyenv install 3.9.12
RUN pyenv global 3.9.12

# Install PyTorch with CUDA support (make sure to adjust this depending on your CUDA version)
RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install pytorch geometric.
RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# Install pytorch3d
RUN pip install fvcore iopath && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code
WORKDIR $CODING_ROOT/code

# Only copy in the source code that is necessary for the dependencies to install
COPY ./taxpose $CODING_ROOT/code/taxpose
COPY ./third_party $CODING_ROOT/code/third_party
COPY ./setup.py $CODING_ROOT/code/setup.py
COPY ./pyproject.toml $CODING_ROOT/code/pyproject.toml
RUN pip install -e .
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

# Set up the entry point
CMD ["python", "-c", "import torch; print(torch.cuda.is_available())"]

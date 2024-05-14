FROM nvidia/cuda:12.2.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Set up the environment.
ENV CODING_ROOT=/opt/baeisner
WORKDIR $CODING_ROOT

# Install the dependencies.
RUN apt-get update && apt-get install -y \
    # Dependencies required for python.
    build-essential \
    curl \
    ffmpeg \
    git \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    libreadline-dev \
    libxml2-dev \
    libxmlsec1-dev \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    # VirtualGL Dependencies.
    libc6 \
    libglu1-mesa \
    libglvnd-dev \
    libxv1 \
    mesa-utils \
    openbox \
    wget \
    xvfb \
    # CoppeliaSim Dependencies.
    libglu1-mesa-dev \
    '^libxcb.*-dev' \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libxrender-dev \
    libx11-xcb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv.
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT=$CODING_ROOT/.pyenv
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install python.
RUN pyenv install 3.9.12
RUN pyenv global 3.9.12

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code
WORKDIR $CODING_ROOT/code

# Download CoppeliaSim
RUN mkdir $CODING_ROOT/.coppelia
WORKDIR $CODING_ROOT/.coppelia
RUN curl -L https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -o CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Set the working directory back to the code directory.
WORKDIR $CODING_ROOT/code

# modify environment variables for coppelia sim
ENV COPPELIASIM_ROOT="$CODING_ROOT/.coppelia/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
ENV QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

# Install VirtualGL
RUN wget --no-check-certificate https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb \
    && dpkg -i virtualgl_*.deb \
    && rm virtualgl_*.deb

# Configure VirtualGL
RUN /opt/VirtualGL/bin/vglserver_config +s +f -t +egl

# Setup environment variables for NVIDIA and VirtualGL
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Copy in the requirements.
COPY requirements-gpu.txt .

RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir wheel==0.40.0

# Install the requirements.
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy in the third-party directory.
COPY third_party third_party

# Install the third-party libraries.
RUN pip install --no-cache-dir -e third_party/ndf_robot

# Install pyrep.
RUN pip install --no-cache-dir --no-build-isolation "pyrep @ git+https://github.com/stepjam/PyRep.git"

# Copy in pyproject.toml.
COPY pyproject.toml .
RUN mkdir taxpose
RUN touch taxpose/py.typed

# Install our project.
RUN pip install --no-cache-dir -e ".[develop,rlbench]"

# Copy in the code.
COPY . .

# Make directories for mounting.
RUN mkdir $CODING_ROOT/data
RUN mkdir $CODING_ROOT/logs

COPY ./docker/entrypoint.sh /opt/baeisner/entrypoint.sh
ENTRYPOINT ["/opt/baeisner/entrypoint.sh"]

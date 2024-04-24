FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set up the environment.
ENV CODING_ROOT=/opt/baeisner
WORKDIR $CODING_ROOT

# Install the dependencies.
RUN apt-get update && apt-get install -y \
    # Dependencies required for python.
    build-essential \
    curl \
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

# Copy in the requirements.
COPY requirements-gpu.txt .

RUN pip install --upgrade pip && pip install wheel==0.40.0

# Install the requirements.
RUN pip install -r requirements-gpu.txt

# Copy in the third-party directory.
COPY third_party third_party

# Install the third-party libraries.
RUN pip install -e third_party/ndf_robot

# Copy in pyproject.toml.
COPY pyproject.toml .
RUN mkdir taxpose
RUN touch taxpose/py.typed

RUN pip install -e ".[develop]"

# Copy in the code.
COPY . .

FROM ubuntu:18.04

LABEL maintainer="matheus.sasso17@gmail.com"

####################################################################
#                       Install Linux Dependencies
####################################################################

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    vim \
    chromium-browser \
    curl \
    libssl-dev \
    git \
    mercurial \
    pepperflashplugin-nonfree \
    libffi-dev \
    wget \
 && rm -rf /var/lib/apt/lists/*
####################################################################

####################################################################
#                       Install Python
####################################################################

# Create a working directory
WORKDIR /app/

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

####################################################################


####################################################################
#                       Install Poetry
####################################################################

USER root

ENV PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    WORKSPACE_TMP="/app/reports" \
    POETRY_VERSION=1.1.11

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && rm -r /var/lib/apt/lists/* \
    && mkdir -p /app

COPY ./setup.py /app/setup.py
COPY ./poetry.lock /app/poetry.lock
COPY ./pyproject.toml /app/pyproject.toml

ARG ENVIRON="production"

# # hadolint ignore=SC2046
RUN poetry config virtualenvs.create false \
    && poetry install \
        $(if [ "$ENVIRON" = 'production' ]; then echo '--no-dev'; fi) \
        --no-interaction --no-ansi

####################################################################


####################################################################
#                       Notebook Configs
####################################################################
RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/
COPY . .

RUN pip3 install jupyter
RUN python3 module.py
WORKDIR /src/notebooks


# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
####################################################################


####################################################################
#                       Final Configs
####################################################################
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
####################################################################

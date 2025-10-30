FROM tensorflow/tensorflow:latest-jupyter

ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${NB_GID} ${NB_USER} || true && \
    useradd -m -u ${NB_UID} -g ${NB_GID} -s /bin/bash ${NB_USER} || true && \
    \
    echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${NB_USER} && \
    chmod 0440 /etc/sudoers.d/${NB_USER}

USER ${NB_USER}

RUN pip install --no-cache-dir \
    s3fs \
    tqdm \
    timezonefinder \
    scikit-learn \
    matplotlib \
    requests

USER root
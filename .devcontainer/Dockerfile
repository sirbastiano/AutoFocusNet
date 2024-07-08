FROM --platform=linux/amd64 ubuntu:22.04
RUN apt-get update && apt-get install -y python3 python3-pip wget git

ENV MINICONDA_VERSION 4.9.2
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install anaconda:
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda update -n base -c defaults conda


RUN $CONDA_DIR/bin/conda create -n sarlib python=3.9
# Combine the environment activation and package installations into one RUN command
RUN echo "source activate sarlib" > ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc \
    && pip install SARLens@git+https://github.com/sirbastiano/AutoFocusNet.git"

# NOTE: intentionally NOT using s6 init as the entrypoint
# This would prevent container debugging if any of those service crash
RUN mkdir /workspace
WORKDIR /workspace
# Auto-setting the kernel
CMD ["/bin/bash", "conda", "install", "-y", "-q", "-n", "simulator", "ipykernel", "--update-deps", "--force-reinstall"]
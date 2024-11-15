# Use Ubuntu as the base image
FROM ubuntu:20.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install some dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh

# Make sure conda is in the path
ENV PATH=/root/miniconda3/bin:$PATH

# Check if conda is successfully installed
RUN conda --version

# Set the working directory to course repo
WORKDIR /home/dyco/work/geometric_computing 
# Copy the environment.yml file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Set up conda environment activation
RUN echo "source activate $(head -n 1 environment.yml | cut -d ' ' -f 2)" > ~/.bashrc

# Ensure the environment is activated
CMD ["/bin/bash"]

# Pull Ubuntu 24.04 image with Nvidia support
FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04

# Update Ubuntu
RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Setup working dir, mounted volume should also point to this
RUN mkdir -p /workspace
WORKDIR /workspace

# Make folders
RUN mkdir Data
RUN mkdir Logs
RUN mkdir Secrets

# Copy files
COPY Data /workspace/Data
COPY Entities /workspace/Entities
COPY Experiments /workspace/Experiments
COPY Logs /workspace/Logs
COPY Managers /workspace/Managers
COPY Processing /workspace/Processing
COPY Assets /workspace/Assets
COPY Secrets /workspace/Secrets
COPY README.md /workspace/
COPY requirements.txt /workspace/

# Create virtual environment
RUN uv venv .venv --python 3.12

# Activate virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"

# Install PyTorch
RUN uv pip install torch torchvision

# Install RAPIDS
RUN uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu13==26.4.*" "dask-cudf-cu13==26.4.*" "cuml-cu13==26.4.*" \
    "cugraph-cu13==26.4.*" "nx-cugraph-cu13==26.4.*" "cuxfilter-cu13==26.4.*" \
    "cucim-cu13==26.4.*" "pylibraft-cu13==26.4.*" "raft-dask-cu13==26.4.*" \
    "cuvs-cu13==26.4.*" "nvforest-cu13==26.4.*" "nx-cugraph-cu13==26.4.*"

# Install packages
RUN uv pip install -r requirements.txt

# Expose marimo port
EXPOSE 2718

# Start marimo server
WORKDIR /workspace/Experiments
CMD ["/workspace/.venv/bin/marimo", "edit", "notebook.py", "--no-token", "--host", "0.0.0.0"]
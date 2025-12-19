# BASE IMAGE
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# SYSTEM DEPENDENCIES
# Cài Python, pip, git
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3 thành python
RUN ln -s /usr/bin/python3 /usr/bin/python

# PROJECT SETUP
# Set working dir
WORKDIR /code

# Copy source code vào container
COPY . /code

# INSTALL LIBRARIES
# Upgrade pip và install từ requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# EXECUTION
# Chạy inference.sh khi container start
CMD ["bash", "inference.sh"]

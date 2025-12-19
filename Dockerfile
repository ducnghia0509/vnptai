# BASE IMAGE
FROM ubuntu:20.04

# SYSTEM DEPENDENCIES
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Link python3 thành python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /code

# COPY FILE - CHỈ copy những file có ở thư mục gốc của bạn
# 1. Copy requirements trước để tận dụng cache Docker
COPY requirements.txt .
# 2. Copy tất cả file code Python, JSON, Markdown, Shell
COPY *.py .
COPY *.json .
COPY *.md .
COPY *.sh .

# INSTALL LIBRARIES
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# EXECUTION
CMD ["bash", "inference.sh"]
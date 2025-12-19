#!/bin/bash
python get_data.py
python data_process.py
python faiss_index.py
# Chạy inference
python main.py
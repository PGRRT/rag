# Installation

### Milvus (Vector database)
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.7/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker compose up -d
```

### Python
**IMPORTANT** Uncomment PyTorch CUDA in the requirements.txt file if you want to use it instead of PyTorch CPU
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
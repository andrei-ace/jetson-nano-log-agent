REMOTE       := andrei@jetson-nano.local
REMOTE_DIR   := /ssd/jetson-log-agent
MODEL_DIR    := /ssd/jetson-log-agent/models
MODEL_NAME   := NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
LLAMA_SERVER := /opt/llama.cpp/build/bin/llama-server

export PATH := .venv/bin:$(HOME)/.local/bin:$(PATH)

SWAP_SIZE    := 8G
SWAP_FILE    := /ssd/swapfile

.PHONY: deploy setup install check-deps hf-login download-model server gen-logs build-index swap help

help:
	@echo "From dev machine:"
	@echo "  make deploy         - Rsync project to Jetson"
	@echo "  make setup          - Deploy + install + download model on Jetson"
	@echo ""
	@echo "On Jetson (cd /ssd/jetson-log-agent):"
	@echo "  make swap           - Create 8G swap on SSD (idempotent)"
	@echo "  make install        - Create venv and install deps"
	@echo "  make download-model - Download Nemotron-3 Nano 4B GGUF"
	@echo "  make server         - Start llama-server (auto-downloads model)"
	@echo "  make gen-logs       - Regenerate logs anchored to current time"
	@echo "  ./launch.sh         - Gen logs + run agent in bwrap sandbox"

# ── Dev machine ─────────────────────────────────────────────────────────────

deploy:
	ssh $(REMOTE) "mkdir -p $(REMOTE_DIR)"
	rsync -avz --delete \
		--exclude '.venv/' \
		--exclude '__pycache__/' \
		--exclude '*.egg-info/' \
		--exclude '.git/' \
		--exclude 'models/' \
		--exclude 'kb_index/' \
		--exclude 'output/' \
		./ $(REMOTE):$(REMOTE_DIR)/

setup: deploy
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && make swap && make install && make download-model'

# ── On Jetson ───────────────────────────────────────────────────────────────

install: check-deps
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv 0.6.x..." && curl -LsSf https://astral.sh/uv/0.6/install.sh | sh; }
	uv venv
	uv pip install --python .venv/bin/python 'langchain-openai>=0.3,<1' 'langgraph>=0.2,<1' 'fastembed>=0.4,<1' 'faiss-cpu>=1.7,<2'

check-deps:
	@missing=""; \
	command -v bwrap >/dev/null 2>&1 || missing="$$missing bubblewrap"; \
	command -v socat >/dev/null 2>&1 || missing="$$missing socat"; \
	{ test -x $(LLAMA_SERVER) || command -v llama-server >/dev/null 2>&1; } || missing="$$missing llama.cpp"; \
	if [ -n "$$missing" ]; then \
		echo "ERROR: Missing dependencies:$$missing"; \
		echo "  sudo apt install bubblewrap socat"; \
		echo "  See https://github.com/ggerganov/llama.cpp for llama.cpp build instructions"; \
		exit 1; \
	fi

HF_MODEL_REV := e0f60ca53a1e42e9a0e36e693690a3ab8a867c90
HF_MODEL_URL := https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/$(HF_MODEL_REV)/$(MODEL_NAME)
# SHA256 of NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf at the pinned revision.
# To update: download the model and run sha256sum on it, then replace this value.
MODEL_SHA256 := SKIP

download-model:
	mkdir -p $(MODEL_DIR)
	@if [ -f $(MODEL_DIR)/$(MODEL_NAME) ]; then \
		echo "Model already downloaded."; \
	elif [ -f ~/.cache/huggingface/token ]; then \
		echo "Downloading with HF token..."; \
		curl -L --progress-bar \
			-H "Authorization: Bearer $$(cat ~/.cache/huggingface/token)" \
			-o $(MODEL_DIR)/$(MODEL_NAME) $(HF_MODEL_URL); \
	else \
		echo "WARNING: No HF token found, download may be slow. Run 'make hf-login' first."; \
		curl -L --progress-bar \
			-o $(MODEL_DIR)/$(MODEL_NAME) $(HF_MODEL_URL); \
	fi
	@if [ "$(MODEL_SHA256)" != "SKIP" ]; then \
		echo "Verifying checksum..."; \
		echo "$(MODEL_SHA256)  $(MODEL_DIR)/$(MODEL_NAME)" | sha256sum -c -; \
	else \
		echo "NOTE: Set MODEL_SHA256 in Makefile to enable integrity check."; \
		echo "  sha256sum $(MODEL_DIR)/$(MODEL_NAME)"; \
	fi

hf-login:
	uv pip install --python .venv/bin/python huggingface-hub
	.venv/bin/python -c "from huggingface_hub import login; login()"

LLAMA_CMD := $(shell test -x $(LLAMA_SERVER) && echo $(LLAMA_SERVER) || command -v llama-server 2>/dev/null)

server: $(MODEL_DIR)/$(MODEL_NAME)
	$(LLAMA_CMD) \
		--model $(MODEL_DIR)/$(MODEL_NAME) \
		--port 8080 \
		--n-gpu-layers -1 \
		--ctx-size 16384 \
		--reasoning-format none

$(MODEL_DIR)/$(MODEL_NAME):
	$(MAKE) download-model

gen-logs:
	python gen_logs.py

build-index:
	python build_index.py

swap:
	@if swapon --show | grep -q $(SWAP_FILE); then \
		echo "Swap already active: $(SWAP_FILE)"; \
	else \
		echo "Creating $(SWAP_SIZE) swap at $(SWAP_FILE)..."; \
		sudo fallocate -l $(SWAP_SIZE) $(SWAP_FILE); \
		sudo chmod 600 $(SWAP_FILE); \
		sudo mkswap $(SWAP_FILE); \
		sudo swapon $(SWAP_FILE); \
		grep -q $(SWAP_FILE) /etc/fstab || echo "$(SWAP_FILE) none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null; \
		echo "Swap enabled (persistent across reboots)"; \
	fi

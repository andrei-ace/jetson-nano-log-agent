REMOTE       := andrei@jetson-nano.local
REMOTE_DIR   := /ssd/jetson-log-agent
MODEL_DIR    := /ssd/jetson-log-agent/models
MODEL_NAME   := NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
LLAMA_SERVER := /opt/llama.cpp/build/bin/llama-server

export PATH := .venv/bin:$(HOME)/.local/bin:$(PATH)

.PHONY: deploy setup install hf-login download-model server gen-logs run help

help:
	@echo "From dev machine:"
	@echo "  make deploy         - Rsync project to Jetson"
	@echo "  make setup          - Deploy + install + download model on Jetson"
	@echo ""
	@echo "On Jetson (cd /ssd/jetson-log-agent):"
	@echo "  make install        - Create venv and install deps"
	@echo "  make download-model - Download Nemotron-3 Nano 4B GGUF"
	@echo "  make server         - Start llama-server (auto-downloads model)"
	@echo "  make gen-logs       - Regenerate logs anchored to current time"
	@echo "  make run            - Gen logs + run agent (no sandbox)"
	@echo "  ./launch.sh         - Run agent in bwrap sandbox"

# ── Dev machine ─────────────────────────────────────────────────────────────

deploy:
	ssh $(REMOTE) "mkdir -p $(REMOTE_DIR)"
	rsync -avz --delete \
		--exclude '.venv/' \
		--exclude '__pycache__/' \
		--exclude '*.egg-info/' \
		--exclude '.git/' \
		--exclude 'models/' \
		./ $(REMOTE):$(REMOTE_DIR)/

setup: deploy
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && make install && make download-model'

# ── On Jetson ───────────────────────────────────────────────────────────────

install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv
	uv pip install --python .venv/bin/python 'langchain-openai>=0.3,<1' 'langgraph>=0.2,<1'

HF_MODEL_URL := https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/$(MODEL_NAME)

download-model:
	mkdir -p $(MODEL_DIR)
	@if [ -f ~/.cache/huggingface/token ]; then \
		echo "Downloading with HF token..."; \
		curl -L --progress-bar \
			-H "Authorization: Bearer $$(cat ~/.cache/huggingface/token)" \
			-o $(MODEL_DIR)/$(MODEL_NAME) $(HF_MODEL_URL); \
	else \
		echo "WARNING: No HF token found, download may be slow. Run 'make hf-login' first."; \
		curl -L --progress-bar \
			-o $(MODEL_DIR)/$(MODEL_NAME) $(HF_MODEL_URL); \
	fi

hf-login:
	uv pip install --python .venv/bin/python huggingface-hub
	.venv/bin/python -c "from huggingface_hub import login; login()"

server: $(MODEL_DIR)/$(MODEL_NAME)
	$(LLAMA_SERVER) \
		--model $(MODEL_DIR)/$(MODEL_NAME) \
		--port 8080 \
		--n-gpu-layers -1 \
		--ctx-size 4096

$(MODEL_DIR)/$(MODEL_NAME):
	$(MAKE) download-model

gen-logs:
	python gen_logs.py

run: gen-logs
	LOG_DIR=./demo_logs python run_agent.py

.PHONY: gpt2-mini-prepare gpt2-medium-prepare gpt2-large-prepare

HF_GPT2_MINI_BASE = https://huggingface.co/erwanf/gpt2-mini/resolve/main
HF_GPT2_MEDIUM_BASE = https://huggingface.co/gpt2-medium/resolve/main
HF_GPT2_LARGE_BASE = https://huggingface.co/gpt2-large/resolve/main

GPT2_MINI_DIR = data/gpt2mini/gpt2-mini
GPT2_MINI_WEIGHTS = data/gpt2mini/model.safetensors
GPT2_MEDIUM_DIR = data/gpt2medium/gpt2-medium
GPT2_MEDIUM_WEIGHTS = data/gpt2medium/model.safetensors
GPT2_LARGE_DIR = data/gpt2large/gpt2-large
GPT2_LARGE_WEIGHTS = data/gpt2large/model.safetensors

gpt2-mini-prepare:
	@mkdir -p $(GPT2_MINI_DIR)
	@echo "Downloading GPT-2 mini files into $(GPT2_MINI_DIR) and data/gpt2mini/..."
	@set -e; \
	fetch() { \
		url="$$1"; \
		dst="$$2"; \
		if [ -f "$$dst" ]; then \
			echo "OK  $$dst"; \
			return; \
		fi; \
		echo "GET $$dst"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L -o "$$dst" "$$url"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -O "$$dst" "$$url"; \
		else \
			echo "Error: need curl or wget to download files"; \
			exit 1; \
		fi; \
	}; \
	fetch "$(HF_GPT2_MINI_BASE)/config.json" "$(GPT2_MINI_DIR)/config.json"; \
	fetch "$(HF_GPT2_MINI_BASE)/vocab.json" "$(GPT2_MINI_DIR)/vocab.json"; \
	fetch "$(HF_GPT2_MINI_BASE)/merges.txt" "$(GPT2_MINI_DIR)/merges.txt"; \
	fetch "$(HF_GPT2_MINI_BASE)/tokenizer_config.json" "$(GPT2_MINI_DIR)/tokenizer_config.json"; \
	fetch "$(HF_GPT2_MINI_BASE)/special_tokens_map.json" "$(GPT2_MINI_DIR)/special_tokens_map.json"; \
	fetch "$(HF_GPT2_MINI_BASE)/generation_config.json" "$(GPT2_MINI_DIR)/generation_config.json"; \
	fetch "$(HF_GPT2_MINI_BASE)/model.safetensors" "$(GPT2_MINI_WEIGHTS)"

gpt2-medium-prepare:
	@mkdir -p $(GPT2_MEDIUM_DIR)
	@echo "Downloading GPT-2 medium files into $(GPT2_MEDIUM_DIR) and data/gpt2medium/..."
	@set -e; \
	fetch() { \
		url="$$1"; \
		dst="$$2"; \
		if [ -f "$$dst" ]; then \
			echo "OK  $$dst"; \
			return; \
		fi; \
		echo "GET $$dst"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L -o "$$dst" "$$url"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -O "$$dst" "$$url"; \
		else \
			echo "Error: need curl or wget to download files"; \
			exit 1; \
		fi; \
	}; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/config.json" "$(GPT2_MEDIUM_DIR)/config.json"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/vocab.json" "$(GPT2_MEDIUM_DIR)/vocab.json"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/merges.txt" "$(GPT2_MEDIUM_DIR)/merges.txt"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/tokenizer_config.json" "$(GPT2_MEDIUM_DIR)/tokenizer_config.json"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/special_tokens_map.json" "$(GPT2_MEDIUM_DIR)/special_tokens_map.json"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/generation_config.json" "$(GPT2_MEDIUM_DIR)/generation_config.json"; \
	fetch "$(HF_GPT2_MEDIUM_BASE)/model.safetensors" "$(GPT2_MEDIUM_WEIGHTS)"

gpt2-large-prepare:
	@mkdir -p $(GPT2_LARGE_DIR)
	@echo "Downloading GPT-2 large files into $(GPT2_LARGE_DIR) and data/gpt2large/..."
	@set -e; \
	fetch() { \
		url="$$1"; \
		dst="$$2"; \
		if [ -f "$$dst" ]; then \
			echo "OK  $$dst"; \
			return; \
		fi; \
		echo "GET $$dst"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L -o "$$dst" "$$url"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -O "$$dst" "$$url"; \
		else \
			echo "Error: need curl or wget to download files"; \
			exit 1; \
		fi; \
	}; \
	fetch "$(HF_GPT2_LARGE_BASE)/config.json" "$(GPT2_LARGE_DIR)/config.json"; \
	fetch "$(HF_GPT2_LARGE_BASE)/vocab.json" "$(GPT2_LARGE_DIR)/vocab.json"; \
	fetch "$(HF_GPT2_LARGE_BASE)/merges.txt" "$(GPT2_LARGE_DIR)/merges.txt"; \
	fetch "$(HF_GPT2_LARGE_BASE)/tokenizer_config.json" "$(GPT2_LARGE_DIR)/tokenizer_config.json"; \
	fetch "$(HF_GPT2_LARGE_BASE)/special_tokens_map.json" "$(GPT2_LARGE_DIR)/special_tokens_map.json"; \
	fetch "$(HF_GPT2_LARGE_BASE)/generation_config.json" "$(GPT2_LARGE_DIR)/generation_config.json"; \
	fetch "$(HF_GPT2_LARGE_BASE)/model.safetensors" "$(GPT2_LARGE_WEIGHTS)"

.PHONY: gpt2-mini-test
gpt2-mini-test:
	go run ./experiments/gpt2mini -prompt 'Hello, Im a language model,'

.PHONY: gpt2-medium-test
gpt2-medium-test:
	go run ./experiments/gpt2medium -prompt 'Hello, Im a language model,'

.PHONY: gpt2-large-test
gpt2-large-test:
	go run ./experiments/gpt2large -prompt 'Hello, Im a language model,'

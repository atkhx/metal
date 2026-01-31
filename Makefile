.PHONY: gpt2-mini-prepare

HF_GPT2_MINI_BASE = https://huggingface.co/erwanf/gpt2-mini/resolve/main
GPT2_MINI_DIR = data/gpt2mini/gpt2-mini
GPT2_MINI_WEIGHTS = data/gpt2mini/model.safetensors

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

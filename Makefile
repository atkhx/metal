.PHONY: gpt2-mini-prepare gpt2-medium-prepare gpt2-large-prepare

HF_GPT2_MINI_BASE = https://huggingface.co/erwanf/gpt2-mini/resolve/main
HF_GPT2_MEDIUM_BASE = https://huggingface.co/gpt2-medium/resolve/main
HF_GPT2_LARGE_BASE = https://huggingface.co/gpt2-large/resolve/main

GPT2_MINI_DIR = data/gpt2mini
GPT2_MEDIUM_DIR = data/gpt2medium
GPT2_LARGE_DIR = data/gpt2large

define fetch_file
	@if [ -f "$(2)" ]; then \
		echo "OK  $(2)"; \
	else \
		echo "GET $(2)"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L -o "$(2)" "$(1)"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -O "$(2)" "$(1)"; \
		else \
			echo "Error: need curl or wget to download files"; \
			exit 1; \
		fi; \
	fi
endef

gpt2-mini-prepare:
	@mkdir -p $(GPT2_MINI_DIR)
	@echo "Downloading GPT-2 mini files into $(GPT2_MINI_DIR)/..."
	$(call fetch_file,$(HF_GPT2_MINI_BASE)/config.json,$(GPT2_MINI_DIR)/config.json)
	$(call fetch_file,$(HF_GPT2_MINI_BASE)/vocab.json,$(GPT2_MINI_DIR)/vocab.json)
	$(call fetch_file,$(HF_GPT2_MINI_BASE)/merges.txt,$(GPT2_MINI_DIR)/merges.txt)
	$(call fetch_file,$(HF_GPT2_MINI_BASE)/model.safetensors,$(GPT2_MINI_DIR)/model.safetensors)

gpt2-medium-prepare:
	@mkdir -p $(GPT2_MEDIUM_DIR)
	@echo "Downloading GPT-2 medium files into $(GPT2_MEDIUM_DIR)/..."
	$(call fetch_file,$(HF_GPT2_MEDIUM_BASE)/config.json,$(GPT2_MEDIUM_DIR)/config.json)
	$(call fetch_file,$(HF_GPT2_MEDIUM_BASE)/vocab.json,$(GPT2_MEDIUM_DIR)/vocab.json)
	$(call fetch_file,$(HF_GPT2_MEDIUM_BASE)/merges.txt,$(GPT2_MEDIUM_DIR)/merges.txt)
	$(call fetch_file,$(HF_GPT2_MEDIUM_BASE)/model.safetensors,$(GPT2_MEDIUM_DIR)/model.safetensors)

gpt2-large-prepare:
	@mkdir -p $(GPT2_LARGE_DIR)
	@echo "Downloading GPT-2 large files into $(GPT2_LARGE_DIR)/..."
	$(call fetch_file,$(HF_GPT2_LARGE_BASE)/config.json,$(GPT2_LARGE_DIR)/config.json)
	$(call fetch_file,$(HF_GPT2_LARGE_BASE)/vocab.json,$(GPT2_LARGE_DIR)/vocab.json)
	$(call fetch_file,$(HF_GPT2_LARGE_BASE)/merges.txt,$(GPT2_LARGE_DIR)/merges.txt)
	$(call fetch_file,$(HF_GPT2_LARGE_BASE)/model.safetensors,$(GPT2_LARGE_DIR)/model.safetensors)

.PHONY: gpt2-mini-test gpt2-medium-test gpt2-large-test
gpt2-mini-test:
	go run ./experiments/gpt2 -model mini -prompt 'Hello, Im a language model,'

gpt2-medium-test:
	go run ./experiments/gpt2 -model medium -prompt 'Hello, Im a language model,'

gpt2-large-test:
	go run ./experiments/gpt2 -model large -prompt 'Hello, Im a language model,'

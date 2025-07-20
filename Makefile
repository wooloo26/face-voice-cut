UV := uv run
RUFF := $(UV) ruff
MYPY := $(UV) mypy
HATCH := $(UV) hatch
PRE_COMMIT := $(UV) pre-commit
CLI := $(UV) cli

SRC_DIR := src
CONFIG_FILE := mypy.ini
TARGET_PATH := C:/Space/downloads/target

.PHONY: fix lint format prepare build generate extract clip

fix:
	$(RUFF) check --fix
	$(RUFF) format

lint:
	$(RUFF) check
	$(MYPY) $(SRC_DIR) --config-file=$(CONFIG_FILE)

format:
	$(RUFF) format

build:
	$(HATCH) build

prepare:
	$(PRE_COMMIT) install

generate:
	$(CLI) generate

extract:
	$(CLI) extract $(TARGET_PATH)

clip:
	$(CLI) clip $(TARGET_PATH)
.PHONY: test lint inspect

test:
	pytest

lint:
	ruff check src tests

inspect:
	ct-restore inspect-model


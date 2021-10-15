tests:
	pytest tests

formatting:
	black vision_transformers/
	isort vision_transformers/

	black tests/
	isort tests/


notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean_log:
	rm -rf lightning_logs

tensorboard:
	tensorboard --logdir ./lightning_logs

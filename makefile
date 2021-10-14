tests:
	pytest tests

formatting:
	black vision_transformers/
	isort vision_transformers/

notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean_log:
	rm -rf lightning_logs

tensorboard:
	tensorboard --logdir ./lightning_logs

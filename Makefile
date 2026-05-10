PYTHON=python

prepare-oracle:
	$(PYTHON) -m src.cli.prepare_data --regime oracle

prepare-asr:
	$(PYTHON) -m src.cli.prepare_data --regime asr --run-asr

train-fold0:
	$(PYTHON) -m src.cli.train_unimodal --branch audio --regime asr --fold 0
	$(PYTHON) -m src.cli.train_unimodal --branch video --regime asr --fold 0
	$(PYTHON) -m src.cli.train_unimodal --branch text --regime asr --fold 0
	$(PYTHON) -m src.cli.train_fusion --mode late --regime asr --fold 0

eval:
	$(PYTHON) -m src.cli.evaluate --suite full --regime asr

export-onnx:
	$(PYTHON) -m src.serving.export_onnx --all

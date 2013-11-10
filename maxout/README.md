Maxing out the digits
=====================

See [http://fastml.com/maxing-out-the-digits/](http://fastml.com/maxing-out-the-digits/) for description.
	
	csv_dataset - a general dataset wrapper, not needed to run the digits example
	digits.yaml - a config file, set your data paths here
	digits_hardcore.yaml - train longer, get better score
	digits_dataset.py - a wrapper class for the digits in CSV format. You need it in your PYTHONPATH.
	predict.py - get predictions from a model
	
Edit `digits.yaml` to set correct paths to training and test files.	
	
How to run on Windows:
	
	cd <project dir>
	set PYTHONPATH=%PYTHONPATH%;.
	set THEANO_FLAGS=device=gpu,floatX=float32
	train.py digits.yaml
	predict.py digits_best.pkl test.csv predictions.csv

How to run on Unix:

	cd <project dir>
	export PYTHONPATH=$PYTHONPATH:.
	export THEANO_FLAGS=device=gpu,floatX=float32
	python train.py digits.yaml
	python predict.py digits_best.pkl test.csv predictions.csv
	
	
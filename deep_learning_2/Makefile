.PHONY: train

data_Q3_2016.zip:
	wget https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q3_2016.zip

data_Q3_2016: data_Q3_2016.zip
	unzip -o $<

data.npz: data_Q3_2016
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=1200 --execute preprocess.ipynb


train.py: train.ipynb
	ipython nbconvert --to python $<

train: data.npz train.py
	python train.py

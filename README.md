# PredCell
This is the code that we are using for the PredCell project.

## Files
- `predcell_train.py`: trains PredCell
- `predcell_subtractive_relu.py`: defines the architecture and its loss function

## How To Run
1. Install any required python modules.
You need scikit-learn, keras, tensorflow, and pytorch. You can install
these like so:
```
pip install scikit-learn
pip install keras
pip install tensorflow
pip install torch torchvision torchaudio
```
2. Make a directory called "runs" (or whatever you want) which has the files for
Tensorboard. Go to the top of `predcell_train.py` and change the run subfolder
to whatever you want.
3. Run tensorboard in another window with `tensorboard --logdir "runs"`,
and run the training program with `python3 predcell_train.py`.

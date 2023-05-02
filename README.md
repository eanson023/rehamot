<div align="center">

# Cross-Modal Retrieval for 3D Human Motion and Text via MildTriple Loss

</div>

## Description
Official PyTorch implementation of the paper [**"Cross-Modal Retrieval for 3D Human Motion and Text via MildTriple Loss"**](http://???) (Under Review).

![rehamot](rehamot.jpg)

## Installation

### Create conda environment

Anaconda is recommended to create this virtual environment.
```bash
conda create python=3.9 --name rehamot
conda activate rehamot
```

And install the following packages:
```bash
pip install torch torchvision
pip install transformers
pip install omegaconf
pip install hydra-core
pip install pandas
pip install einops
pip install rich
pip install tensorboard tensorboardX tensorboard_logger
pip install matplotlib
```

### Download the datasets

We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).  
Please note that the Humanml3d dataset needs to be processed through that repository, while KIT-ML can be downloaded directly from the [link](https://drive.google.com/drive/folders/1MnixfyGfujSP-4t8w_2QvjtTVpEKr97t?usp=sharing) in its README

Please move the processed or unzip files to the datasets folder under this project, the file directory should look like this:
```bash
./dataset/
./dataset/HumanML3D/
./dataset/KIT-ML/
...
```

### Download text model dependencies

Download distilbert from __Hugging Face__

```bash
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

## Pre-trained Models

### Download models
Download models [here](http://). Unzip and place them under checkpoint directory, which should be like:
```bash
./checkpoints/HumanML3D
./checkpoints/KIT-ML/
```

### Evaluate pre-trained models
To evaluate Rehamot, you must run:
```bash
python evaluate.py folder=FOLDER
```
The ```FOLDER``` can be replaced with specific experiment, such as ```./checkpoints/HumanML3D/mildtripleloss```

### View the pre-training results
We recorded some metrics using tensorboard, which you can view by running the following command:
```bash
tensorboard --logdir=FOLDER
```
The ```FOLDER``` can be replaced with the main experiment, such as ```./checkpoints/HumanML3D```

## Training new models

### Run
The command to launch a training experiment is the folowing:
```bash
python train.py [OPTIONS]
```
You can override anything in the configuration by passing arguments like `foo=value` or `foo.bar=value`. Of course, you can also modify the configuration directly in the configs folder.


### Some optional parameters
#### Datasets
- ``data=human-ml-3d``: Training Rehamot on HumanML3D
- ``data=kit-ml``: Training Rehamot on KIT-ML

#### Threshold
- ``model.threshold_hetero=0.7``: Set the threshold for heteromorphism
- ``model.threshold_homo=0.9``: Set the threshold for homomorphism 

#### Training
- ``machine.device=gpu``: training with CUDA, on an automatically selected GPU (default)
- ``machine.device=cpu``: training on the CPU (not recommended)

If you want to get started with the training model quickly, you can refer to the configuration of the `.hydra/config.yaml` file under the pre-training model provided by us

## Citation
If you find this code to be useful for your research, please consider citing.
```
```

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

[TEMOS](https://github.com/Mathux/TEMOS), [vse++](https://github.com/fartashf/vsepp).

## License

This code is distributed under an [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
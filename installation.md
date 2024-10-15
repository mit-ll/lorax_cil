# Installation

1. Set up your conda environment lorax_cil with Python version 3.8.17
```
conda create --name lorax_cil python==3.8.17
```

2. Activate your conda environment
```
conda activate lorax_cil
```

3. Use pip to install required packages (listed in requirements.txt)
```
pip install -r requirements.txt
```

4. Use pip to install the PyTorch Image Model at commit b87d98b
```
 pip install git+https://github.com/huggingface/pytorch-image-models.git@b87d98b
```

You're ready to go!

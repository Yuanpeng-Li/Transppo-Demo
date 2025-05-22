# TransPPO


## Getting Started


### Installation

Download the ZIP archive and unpack it:

```bash
# Download the repository archive as a ZIP file
curl -L https://anonymous.4open.science/api/repo/Transppo-Demo/zip -o Transppo-Demo.zip

# Create a directory to hold the project
mkdir Transppo-Demo

# Unzip the archive into the newly created directory
unzip Transppo-Demo.zip -d Transppo-Demo

# Change into the project directory
cd Transppo-Demo
```

Create a conda environment with Python 3.10.16 and install dependencies:

```bash
conda create -n transppo_env python=3.10.16
conda activate transppo_env
pip install -r requirements.txt
```

### Training
To train the TransPPO model, run the following command:
```bash
python main/train_TransPPO.py
```
To train the SinglePPO model, run the following command:
```bash
python main/train_SinglePPO.py
```
To train the MultiPPO model, run the following command:
```bash
python main/train_MultiPPO.py
```
### Logs
To view the training logs, run the following command:
```bash
tensorboard --logdir=main/logs/
```

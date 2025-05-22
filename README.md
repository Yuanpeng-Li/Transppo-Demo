# TransPPO


## Getting Started


### Installation


Clone the repository:

```bash
git clone https://github.com/your-username/Transppo-Demo.git
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

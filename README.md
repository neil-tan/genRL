[![bMQUG.gif](https://s6.gifyu.com/images/bMQUG.gif)](https://wandb.ai/neiltan/genRL_cartpole_tune/runs/vcotwkym?nw=nwuserneiltan)

## Installation
We will install Miniconda as a part of Pyenv, and create a virtual environment for the package.
### Install dependencies
#### Ubuntu
```bash
sudo apt-get update
sudo apt install ffmpeg openssl libreadline-dev libsqlite3-dev xz-utils zlib1g-dev tcl-tk tcl-dev tk-dev
```

#### MacOS
```bash 
brew update
brew install ffmpeg openssl readline sqlite3 xz zlib tcl-tk
```

### Install [Pyenv](https://github.com/pyenv/pyenv)
```bash
brew install pyenv
```
Follow the [instructions here](https://github.com/pyenv/pyenv?tab=readme-ov-file#b-set-up-your-shell-environment-for-pyenv) to add pyenv to your shell profile. 

Reload your shelldef or restart your terminal
```bash
exec "$SHELL"
```

### Miniconda
Identify a version of miniconda that you'd like to install. The latest version is recommended.
```bash
# Check the latest version of Miniconda 
pyenv install --list
```
For example, if you want to install Miniconda 3.10.25-1:
```bash
pyenv install miniconda3-3.10-25.1.1-2
```
Activate the miniconda and create a virtual environment:
```bash
pyenv virtualenv miniconda3-4.7.12 genrl
pyenv activate genrl
# to install python 3.12, run:
# conda install python=3.12
```

### Install the package
```bash
git clone git@github.com:neil-tan/genRL.git
cd genRL
# Install the package
pip install -e .
```

### Run Example
A simple Cartpole PPO example:
```bash
conda activate genRL
python examples/minimalRL_cartpole_ppo.py
```

Hyperparameter sweeping example:
```bash
python examples/tune_ppo.py
```
[![bMW2o.png](https://s6.gifyu.com/images/bMW2o.png)](https://wandb.ai/neiltan/genRL_cartpole_tune?nw=nwuserneiltan)

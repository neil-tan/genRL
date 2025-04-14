[![bMQUG.gif](https://s6.gifyu.com/images/bMQUG.gif)](https://wandb.ai/neiltan/genRL_cartpole_tune/runs/vcotwkym?nw=nwuserneiltan)

## Installation
Install the prerequisites, create a virtual environment using minconda and pyenv. Click on the tabs below to see the instructions for your OS.
<details>
  <summary>Linux</summary>
We will install Miniconda as a part of Pyenv, and create a virtual environment for the package.

#### Install dependencies

```bash
sudo apt-get update
sudo apt install ffmpeg openssl libreadline-dev libsqlite3-dev xz-utils zlib1g-dev tcl-tk tcl-dev tk-dev
```

#### Install [Pyenv](https://github.com/pyenv/pyenv)

```bash
curl -fsSL https://pyenv.run | bash
```

Follow the [instructions here](https://github.com/pyenv/pyenv?tab=readme-ov-file#b-set-up-your-shell-environment-for-pyenv) to add pyenv to your shell profile. 

Reload your shelldef or restart your terminal.
```bash
exec "$SHELL"
```


#### Miniconda
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
</details>

<details>
  <summary>MacOS</summary>

#### Install dependencies
```bash 
brew update
brew install ffmpeg openssl readline sqlite3 xz zlib tcl-tk
```
#### Install [Pyenv](https://github.com/pyenv/pyenv)
```bash
brew install pyenv
```
Follow the [instructions here](https://github.com/pyenv/pyenv?tab=readme-ov-file#b-set-up-your-shell-environment-for-pyenv) to add pyenv to your shell profile. 

Reload your shelldef or restart your terminal
```bash
exec "$SHELL"
```
#### Miniconda

The Miniconda package isn't available on Pyenv for MacOS, so we will install it manually.
- Install [Miniconda](https://docs.anaconda.com/miniconda/install)
```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
- Ensure Miniconda is in your PATH
```bash
# Replace <PATH-TO-CONDA> with the file path to your conda installation
# <PATH-TO-CONDA>/bin/conda init zsh
~/minianaconda3/bin/conda init zsh
```
- If you have Pyenv on your system, it's a good idea to set conda auto-activate to false
```bash
conda config --set auto_activate_base <TRUE_OR_FALSE>
```
- Create a new conda environment and install dependencies
```bash
conda create --name genRL python=3.12 -y
conda activate genRL
```

</details>

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
python examples/train.py algo:ppo-config --algo.n_epi 185
```

Hyperparameter sweeping example:
```bash
python examples/tune_ppo.py
```
[![bMW2o.png](https://s6.gifyu.com/images/bMW2o.png)](https://wandb.ai/neiltan/genRL_cartpole_tune?nw=nwuserneiltan)

### Related Links
- [minimal RL](https://github.com/seungeunrho/minimalRL)
- [cleanRL](https://github.com/vwxyzjn/cleanrl)

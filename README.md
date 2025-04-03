## Installation
Please stick to the Miniconda installation for now. Pyenv is not working on Mac.
- Install dependencies with [Homebrew](https://brew.sh/)
```bash 
brew update
brew install ffmpeg openssl readline sqlite3 xz zlib tcl-tk
```

### Miniconda
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
pip install -e .
```

#### Managing Conda Environments
- List all conda environments
```bash
conda env list
```
- Remove a conda environment (Note: you cannot use pyenv to remove conda environments)
```bash
conda remove --name environment_name --all
```
- For environments in non-standard locations (like miniconda-latest/envs/gr00t-test), use:
```bash
conda env remove -p path/to/environment
# Example:
# conda env remove -p ~/miniconda-latest/envs/gr00t-test
```

### Pyenv
** Not currently working on Mac. **
- Install [Pyenv](https://github.com/pyenv/pyenv)
```bash
brew install pyenv
# Add the following to your shell profile (~/.zshrc, ~/.bashrc, etc.)
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
# reload shell
exec "$SHELL"
# install python 3.12.9 with tk support
```
Execute the following command (**in a single line**) to install python 3.12.9 with tk support
```bash
env LDFLAGS="-L$(brew --prefix openssl@1.1)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix sqlite3)/lib -L$(brew --prefix xz)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix tcl-tk)/lib" \
CPPFLAGS="-I$(brew --prefix openssl@1.1)/include -I$(brew --prefix readline)/include -I$(brew --prefix sqlite3)/include -I$(brew --prefix xz)/include -I$(brew --prefix zlib)/include -I$(brew --prefix tcl-tk)/include" \
PKG_CONFIG_PATH="$(brew --prefix openssl@1.1)/lib/pkgconfig:$(brew --prefix readline)/lib/pkgconfig:$(brew --prefix sqlite3)/lib/pkgconfig:$(brew --prefix xz)/lib/pkgconfig:$(brew --prefix zlib)/lib/pkgconfig:$(brew --prefix tcl-tk)/lib/pkgconfig" \
pyenv install 3.12.9
```
*** note: the above tk-python support commands are taken from [here](https://dev.to/xshapira/using-tkinter-with-pyenv-a-simple-two-step-guide-hh5) ***

Use `pyenv global 3.12.9` to set the global python version
Or, use `pyenv local 3.12.9` to set the local python version

Setup Pyenv virtual environment and install dependencies
```bash
python --version # should be 3.12.9
pyenv virtualenv genrl && pyenv activate genrl
pip install -e .
```

### Run Example
```bash
conda activate genRL
python examples/minimalRL_cartpole_ppo.py
```

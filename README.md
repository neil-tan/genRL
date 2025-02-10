## Installation
- Install dependencies with [Homebrew](https://brew.sh/)
```bash 
brew update
brew install ffmpeg openssl readline sqlite3 xz zlib tcl-tk
```
- Install [Pyenv]()
```bash
brew install pyenv
# make sure to add the following to your shell profile
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# reload shell
exec "$SHELL"
# install python 3.12.9 with tk support
```
Execute the following command (in a single line) to install python 3.12.9 with tk support
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
pyenv virtualenv genrl && pyenv activate genrl
pip install -e .
```

## Download Assets
```bash
wget https://github.com/MattDerry/pendulum_3d/raw/refs/heads/master/urdf/pendulum.urdf -P assets/urdf
```
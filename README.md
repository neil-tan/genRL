## Installation
Setup Pyenv virtual environment and install dependencies
```bash
brew install ffmpeg
pyenv virtualenv genrl && pyenv activate genrl
pip install -e .
```

## Download Assets
```bash
wget https://github.com/MattDerry/pendulum_3d/raw/refs/heads/master/urdf/pendulum.urdf -P assets/urdf
```
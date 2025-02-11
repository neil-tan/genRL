from setuptools import setup, find_packages

setup( 
    name='genRL', 
    version='0.0', 
    description='RL Blitz', 
    author='Neil Tan', 
    author_email='neil@utensor.ai',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[ 
        'torch',
        'torchvision', 
        'torchaudio',
        'pyrender',
        'genesis-world',
        'pytest',
    ],
)
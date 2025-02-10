from setuptools import setup 

setup( 
	name='genRL', 
	version='0.0', 
	description='RL Blitz', 
	author='Neil Tan', 
	author_email='neil@utensor.ai', 
	# packages=["thirdparty"], 
	install_requires=[ 
	'PyGEL3D==0.1.0',
    'torch',
    'torchvision',
    'torchaudio',
    'genesis-world',
    'pytest',
	], 
#   extras_require={
#     'extra': ['lightning-transformers']
#   }
) 
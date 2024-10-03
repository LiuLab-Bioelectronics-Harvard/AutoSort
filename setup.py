from setuptools import setup, find_packages

with open('README.md','r') as f:
    description = f.read()

setup(
    name='autosort_neuron',
    version='0.0.1.3',
    packages=find_packages(),
    install_requires=[
        'spikeinterface[full,widgets]',
        'torch',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'scipy',
        'networkx',
    ],
    
    long_description=description,
    long_description_content_type='text/markdown',

)
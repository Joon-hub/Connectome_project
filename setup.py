from setuptools import setup, find_packages

setup(
    name='brain_connectivity_pipeline',
    version='0.0.1',
    description='Brain region classification from connectivity data',
    author='Sudhir Joon',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.3',
        'seaborn>=0.11.1',
        'nilearn>=0.6.2',
        'pyyaml>=5.4.1'
    ],
    python_requires='>=3.11'
)

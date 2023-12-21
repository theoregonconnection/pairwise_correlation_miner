from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pairwise_correlation_miner',
    version='0.1',
    packages=find_packages(),
    description='A python function to mine pairwise correlations from a dataframe with many features and targets. Exports the results to a CSV.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael McNamara',
    author_email='theoregonconnection@gmail.com',
    url='https://github.com/theoregonconnection/pairwise_correlation_miner',
    install_requires=[
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.0',
        # 'datetime>=4.3', # This might not be necessary
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

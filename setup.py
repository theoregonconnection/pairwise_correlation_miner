from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pairwise_correlation_miner',
    version='3.2',
    packages=find_packages(),
    description='A python function to mine pairwise correlations from a dataframe with many features and targets. Exports the results to a CSV.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Michael McNamara',
    author_email='theoregonconnection@gmail.com',
    url='https://github.com/theoregonconnection/pairwise_correlation_miner',
    install_requires=[
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.0' 
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    #include_package_data=True,  # Whether to include non-code files from MANIFEST.in
    zip_safe=False
)

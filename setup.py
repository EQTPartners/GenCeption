from setuptools import setup, find_packages

setup(
    name='genception',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.37.1',
        'pillow',
        'requests',
        'scikit-learn',
        'nltk',
        'openai',
        'sentencepiece'
    ],
)

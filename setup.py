from setuptools import setup, find_packages

setup(
    name='NeRo',
    version='0.1.0',
    url='https://github.com/romesco/nero',
    author='Rosario Scalise',
    author_email='rosario@cs.washington.edu',
    description='Elegantly train robots to do complex tasks.',
    packages=find_packages('nero.*'),    
    install_requires=[],
)

from setuptools import find_packages, setup

setup(
    name='jacovid',
    version="0.0.1",
    description='jacques covid',
    packages=find_packages(include=['jacovid', 'jacovid.*']),
    url='https://github.com/reichlab/jacques-covid',
    license='MIT'
)
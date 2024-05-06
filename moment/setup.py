from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="moment",
    version="0.1",
    description="MOMENT: A Family of Open Time-series Foundation Models",
    author="XXXX-1, XXXX-3, XXXX-5, XXXX-7, XXXX-13, Artur Dubrawski",
    author_email="XXXX-2@andrew.XXXX-11.edu",
    license="MIT",
    url="XXXX",
    zip_safe=False,
    packages=find_packages(exclude=['scripts', 'tests']),
    install_requires=required
)

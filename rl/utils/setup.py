from setuptools import setup, find_packages

setup(
    name='longshot-rl-utils',
    version='0.1.0',
    author='bluebread',
    author_email='hotbread70127@gmail.com',
    description='Reinforcement learning utilities for the gym-longshot project',
    packages=find_packages(where='.'),
    install_requires=[
        'pika>=1.3.2',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
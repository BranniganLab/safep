from setuptools import setup

setup(
    name='safep',
    version='0.1.3',
    description='Tools for Analyzing and Debugging (SA)FEP calculations',
    url='https://github.com/BranniganLab/safep',
    author='Brannigan Lab',
    author_email='grace.brannigan@rutgers.edu',
    packages=['safep'],
    install_requires=['numpy>=1.22.0','pandas>=1.4.0', 'alchemlyb==2.0.0', 'matplotlib>=3.5.0', 'natsort>=7.1.0', 'tqdm'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
)

from setuptools import setup

setup(
    name='safep',
    version='0.1.0',    
    description='Tools for Analyzing and Debugging (SA)FEP calculations',
    url='https://github.com/BranniganLab/safep',
    author='Brannigan Lab',
    author_email='grace.brannigan@rutgers.edu',
    packages=['safep'],
    install_requires=['numpy','pandas', 'alchemlyb', 'matplotlib', 'natsort', 'glob'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
)
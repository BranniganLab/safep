from setuptools import setup

setup(
    name='SAFEP',
    version='0.1.0',    
    description='Tools for Analyzing and Debugging (SA)FEP calculations',
    url='https://github.com/BranniganLab/safep',
    author='Brannigan Lab',
    author_email='grace.brannigan@rutgers.edu',
    license='BSD 2-clause',
    packages=['SAFEP'],
    install_requires=['numpy',],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
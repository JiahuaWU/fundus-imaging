from setuptools import setup

setup(
    name='zeiss_umbrella',
    version='0.1.0',
    description='Initial umbrella module for the Zeiss Interpretability and robustness projects',
    url='https://github.com/shuds13/pyexample',
    author='LIONS',
    author_email='',
    license='Apache 2.0',
    packages=['zeiss_umbrella'],
    install_requires=['matplotlib',
                      'numpy',
                      'torch',
                      'seaborn',
                      'scipy',
                      'sacred',
                      'attrs',
                      'pymongo', # for sacred
                      'tinydb', # for sacred TinyDB
                      'tinydb-serialization', # for sacred sqlite
                      'hashfs', # for sacred sqlite
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

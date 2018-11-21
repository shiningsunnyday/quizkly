"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding


setup(
    name='revup-core',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='RevUP Core Algorithms',
    # long_description=long_description,

    # url='https://revup.temp',

    author='RevUP',
    # author_email='revup@gmail.com',

    # license='',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='revup',

    packages=['revup',
              'revup.core',
              'revup.core.DistractorSelection',
              'revup.core.GapSelection',
              'revup.core.TopicModelling',
              'revup.core.Utils'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'theano',
        'gensim',
        'nltk',
        'scikit-learn',
        #'pattern',
        'kenlm',
        'spacy',
        'marisa-trie',
        'xgboost'
        ],
    dependency_links=[
        'git+https://gitlab.com/revup/practnlptools.git#egg=practNLPTools-1.0',
        'git+https://github.com/kpu/kenlm.git#egg=kenlm-1.0'
        ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'gui': ['easygui']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'revup.core': [
            # 'data/datacbv3.txt',
            'data/stopwords.txt',
            # 'models/[1000, 500, 500, 128],20,100,0.0001,30,dbn-autocb.pkl',
            # 'models/arpalm.lm',
            # 'models/dl30bio.pkl',
            # 'models/gaps.gsm',
            # 'models/lm.pkl',
            # 'models/qwmdatabio70.wm',
            # 'models/qwmdatabio70.wm.syn0.npy',
            # 'models/wmdatabio70.wm',
            # 'models/wmdatabio70.wm.syn0.npy',
            # 'models/wmdatabio70.wm.syn1.npy'],
        ]
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'revup-core = revup.core.main:main',
        ],
    },
)

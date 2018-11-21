# Basic Setup
Based off CentOS 7 environment, but different distros have different package names

Install `epel-release` to access additional packages

Install yum packages:
* gcc-c++
* lapack
* lapack-devel
* openblas
* openblas-devel
* libxml2
* libxml2-devel
* libxslt
* libxslt-devel
* python
* python-devel
* python-pip

(For building wheels)
* wheel
* postgresql-devel

(Optional utilities)
* git
* openssh-client
* tmux

Copy in code repos, via
* git clone
* Docker volume link
* etc

Install python dependencies with:
(for web repo)
pip install -r requirements.txt
(for core repo, installed in editable mode)
pip install -e .

Install nltk dependencies:
* wordnet
* maxent\_treebank\_pos\_tagger
(Run `python -m nltk.downloader wordnet maxent_treebank_pos_tagger`)

Make local copy of settings file in local/ directory (based off settings/dev.py, prod.py or docker.py)
Customize settings (e.g. database, redis/rabbitmq servers, etc)
Make local copy of run file in scripts/ directory (name it scripts/run, using scripts/run.template as template)

Migrate/setup DB with `scripts/run python manage.py migrate`

Start celery worker (scripts/run-celery)
Start web server (scripts/run-web)

Note: We use a fixedup version of practNLPTools, so pip needs the `--process-dependency-links` flag when installing/building wheel for core repo
Might need `--allow-external practNLPTools --allow-unverified practNLPTools` as well


# Other servers
If using a PostgreSQL database, you can run it either on the same or separate PC/VM/Docker container
Same goes for Redis and RabbitMQ


# Wheels
Wheels are prebuilt python packages, which saves numpy/scipy/etc compile time

`pip wheel` takes similar args as `pip install`, so either specify package name or root dir, or `-r <requirements file>`.
(web repo is not a package, but provides a requirements.txt)

`pip wheel` builds a package and its dependencies each into a .wheel file
Then, `pip install` can be pointed to those wheel files with `-f <wheel directory>`

To skip building dependencies, add `--no-deps`


# Production
_todo: Let's Encrypt, nginx, uwsgi_

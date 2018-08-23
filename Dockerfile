FROM ubuntu:xenial

USER root

# 1. OS
# Retrieve new lists of packages
RUN apt-get -qq update

# 2. APT
# Install g++, nano, pigz, wget
RUN apt-get -qq update \
        && apt-get install -y g++ nano pigz wget \
        && apt-get clean

# 3. JAVA
# Install java
RUN apt-get -qq update && apt-get install -y openjdk-8* && apt-get clean

# 4. PYTHON+PIP
# Install python, python-dev
RUN apt-get -qq update \
    && apt-get install -y apt-utils vim curl apache2 apache2-utils \
    python3-dev python3 libapache2-mod-wsgi-py3 python3-pip \
    apt-transport-https debconf-utils locales unixodbc unixodbc-dev \
	tdsodbc freetds-common freetds-bin freetds-dev \
    && apt-get clean

RUN rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN ln /usr/bin/python3 /usr/bin/python
RUN apt-get -y install python3-pip
RUN ln /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

## Copy over and install the requirements
COPY ./app/requirements.txt /var/www/apache-flask/app/requirements.txt
RUN pip install -r /var/www/apache-flask/app/requirements.txt

# 5. SPARK
# Download Apache Spark ver. 2.1.0 (2016-12-28) to '/tmp' directory
ENV URL_SCHEME=http
ENV URL_NETLOC=d3kbcqa49mib13.cloudfront.net
ENV URL_PATH=/spark-2.2.0-bin-hadoop2.7.tgz
ENV URL=$URL_SCHEME://$URL_NETLOC$URL_PATH
RUN wget --directory-prefix /tmp $URL
# Unpack '/tmp/spark-2.1.0-bin-hadoop2.7.tgz' archive
RUN unpigz --to-stdout /tmp/spark-2.2.0-bin-hadoop2.7.tgz \
        | tar --extract --file - --directory /usr/local/src
# Remove '/tmp/spark-2.1.0-bin-hadoop2.7.tgz' archive
RUN rm /tmp/spark-2.2.0-bin-hadoop2.7.tgz
# Set up Apache Spark
ENV SPARK_HOME=/usr/local/src/spark-2.2.0-bin-hadoop2.7
ENV PYTHON_DIR_PATH=$SPARK_HOME/python/
ENV PY4J_PATH=$SPARK_HOME/python/lib/py4j-0.10.4-src.zip
ENV PYTHONPATH=$PYTHON_DIR_PATH:$PY4J_PATH
COPY log4j.properties $SPARK_HOME/conf/log4j.properties
COPY spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

# Setup application
# Copy over the apache configuration file and enable the site
COPY ./apache-flask.conf /etc/apache2/sites-available/apache-flask.conf
RUN a2ensite apache-flask
RUN a2enmod headers

## Copy over the wsgi file and app contents
COPY ./apache-flask.wsgi /var/www/apache-flask/apache-flask.wsgi
COPY ./run.py /var/www/apache-flask/run.py
COPY ./app /var/www/apache-flask/app/

RUN a2dissite 000-default.conf
RUN a2ensite apache-flask.conf

RUN chmod a+rwx -R /var/www/apache-flask/app/

EXPOSE 80

WORKDIR /var/www/apache-flask/

CMD ["apache2ctl", "-D", "FOREGROUND"]
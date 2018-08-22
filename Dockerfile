FROM ubuntu:latest
MAINTAINER Andre Lamurias <alamurias@lasige.di.fc.ul.pt>
WORKDIR /
COPY bin/ bin/

RUN apt-get update -y
RUN apt-get dist-upgrade -y

# Install Python 3.6
RUN apt-get install software-properties-common -y


# Install Java
RUN \
  echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  apt-get update && apt-get install software-properties-common -y && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update && \
  apt-get install -y oracle-java8-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk8-installer


# Define Commonly Used JAVA_HOME Variable
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

RUN apt-get update && apt-get install unrar
WORKDIR /bin

RUN apt-get update && apt-get install unzip
WORKDIR /bin

RUN apt-get update && apt-get install wget
WORKDIR /bin


# Get Stanford NER 3.5.2
RUN wget http://nlp.stanford.edu/software/stanford-ner-2015-04-20.zip && unzip stanford-ner-2015-04-20.zip
WORKDIR stanford-ner-2015-04-20


# Get Stanford CORENLP
WORKDIR /bin
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip && unzip stanford-corenlp-full-2015-12-09.zip
WORKDIR stanford-corenlp-full-2015-12-09


# Install Genia Sentence Splitter (requires ruby and make)
WORKDIR /bin
RUN apt-get update &&  apt-get install -y ruby
RUN wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && \
    tar -xvzf geniass-1.00.tar.gz && \
    rm geniass-1.00.tar.gz
WORKDIR /bin/geniass
RUN apt-get update -y && apt-get install -y build-essential g++ make && make

WORKDIR bin/
RUN wget https://files.pythonhosted.org/packages/db/ee/087a1b7c381041403105e87d13d729d160fa7d6010a8851ba051b00f7c67/jsre-1.1.0.zip && unzip jsre-1.1.0.zip
WORKDIR jsre


# For Stanford CoreNLP
EXPOSE 9000
WORKDIR /bin/stanford-corenlp-full-2015-12-09
ENV CLASSPATH="`find . -name '*.jar'`"
RUN nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 &


# Install Python Libraries
WORKDIR /
RUN apt-get update -y && apt-get -y install git liblapack-dev liblapack3 libopenblas-base libopenblas-dev
RUN apt-get update -y && apt-get -y install python3-dev libmysqlclient-dev -y
COPY requirements.txt /
RUN apt-get update -y && apt-get install python3-pip -y && pip3 install mysqlclient && apt-get install python3-scipy -y && pip3 install -r requirements.txt


# Copy Repository Dirs and Create Sample Data
WORKDIR /
COPY src/ src/
COPY corpora/ corpora/
COPY data/ data/
COPY models/ models/
COPY GSC+.rar / 
RUN unrar x GSC+.rar
RUN shuf -zn68 -e Annotations/* | xargs -0 mv -t corpora/hpo/test_ann/
RUN for file in corpora/hpo/test_ann/*; do mv Text/"$(basename "$file")" corpora/hpo/test_corpus/; done
RUN mv Annotations/* corpora/hpo/train_ann/
RUN mv Text/* corpora/hpo/train_corpus/
RUN rm -rf Annotations Text log.txt
RUN pip3 install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm


# Initial Configuration
RUN pip3 install --upgrade cython
RUN pip3 install word2vec
RUN python3 -m nltk.downloader punkt
RUN pip3 install python-levenshtein
RUN pip3 install numpy --upgrade
RUN mv bin/base.prop bin/stanford-ner-2015-04-20/
COPY settings_base.json /
COPY settings.json /
ENV RUBYOPT="-KU -E utf-8:utf-8"


# Define Default Command
ENTRYPOINT bash

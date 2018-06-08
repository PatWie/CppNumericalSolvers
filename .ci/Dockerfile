FROM ubuntu:16.04
MAINTAINER Patwie <mail@patwie.com>

RUN apt-get -qq update && apt-get -qq dist-upgrade && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    pkg-config \
    g++ \
    g++-4.9 \
    g++-5 \
    pkg-config \
    zip \
    zlib1g-dev \
    unzip \
    && apt-get -qq clean

WORKDIR /bazel
RUN wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh
RUN wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh.sha256
RUN sha256sum -c bazel-0.14.0-installer-linux-x86_64.sh.sha256
RUN chmod +x bazel-0.14.0-installer-linux-x86_64.sh
RUN ./bazel-0.14.0-installer-linux-x86_64.sh --user
ENV PATH="$PATH:$HOME/bin"

RUN apt-get install ca-certificates -qq -y
RUN apt-get install python -qq -y
WORKDIR /project

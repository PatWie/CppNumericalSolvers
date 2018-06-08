FROM ubuntu:18.04
MAINTAINER Patwie <mail@patwie.com>

RUN apt-get -qq update && apt-get -qq dist-upgrade && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    pkg-config \
    g++ \
    pkg-config \
    zip \
    zlib1g-dev \
    unzip \
    python \
    ca-certificates \
    && apt-get -qq clean

WORKDIR /bazel
RUN wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh
RUN wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh.sha256
RUN sha256sum -c bazel-0.14.0-installer-linux-x86_64.sh.sha256
RUN chmod +x bazel-0.14.0-installer-linux-x86_64.sh
RUN ./bazel-0.14.0-installer-linux-x86_64.sh --user
ENV PATH="$PATH:$HOME/bin"

WORKDIR /project

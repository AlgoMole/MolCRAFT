#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]
  then echo "Please run as root or sudo"
  exit
fi

if ! [ -x "$(command -v docker)" ]; then
  echo 'docker is not installed. installing docker now...'
  curl -fsSL https://get.docker.com -o ~/get-docker.sh; bash ~/get-docker.sh
fi

if ! [ -x "$(command -v nvidia-container-toolkit)" ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && nvidia-ctk runtime configure --runtime=docker \
    && systemctl restart docker
fi


if [ -z "$SUDO_USER" ]; then
    #read user from input
    read -p "Please enter the username of the user you want to add to the docker group: " SUDO_USER
fi

if groups "$SUDO_USER" | grep -q "\bdocker\b"; then
    echo ""
else
    usermod -aG docker $SUDO_USER
    newgrp docker
fi
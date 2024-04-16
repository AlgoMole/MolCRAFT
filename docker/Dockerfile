ARG BASE_IMAGE=${BASE_IMAGE:-null}
ARG PUBLIC_IMAGE=${PUBLIC_IMAGE:-null}
FROM ${BASE_IMAGE} as public

COPY ./asset /tmp/build

RUN apt update && \
    xargs apt install -y < /tmp/build/apt_packages.txt

RUN pip install -r /tmp/build/requirements.txt \
    && rm -rf /tmp/build

##############################################
FROM ${PUBLIC_IMAGE} as private
ARG TORCH_HOME
ARG _USER
ARG _UID
ARG _GID
ARG PW
ARG _HOME=/sharefs/${_USER}

COPY ./asset /tmp/build
RUN xargs apt install -y < /tmp/build/apt_packages.txt
RUN pip install -r /tmp/build/requirements.txt

# Option1: Using unencrypted password/ specifying password
RUN usermod --password $(echo ${PW} | openssl passwd -1 -stdin) root
RUN mkdir -p ${_HOME}
RUN useradd -m ${_USER} -d ${_HOME} --uid=${_UID} -s /usr/bin/zsh && echo "${_USER}:${PW}" | chpasswd

# make the color of zsh-autosuggestions right
ENV TERM xterm-256color

RUN cp -r /root/.oh-my-zsh ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.oh-my-zsh &&\
    cp /root/.zshrc ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.zshrc &&\
    cp /root/.tmux.conf ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.tmux.conf && \
    cp -r /root/.tmux ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.tmux

RUN chown ${_USER}:${_USER} -R /tmp/build && \
    chown ${_USER}:${_USER} -R ${_HOME}
USER ${_UID}:${_GID}
WORKDIR ${_HOME}

ENV TORCH_HOME ${TORCH_HOME}

COPY --chown=${_USER}:${_USER} ./asset/ssh .ssh
RUN chmod 0700 .ssh && chmod 600 .ssh/id_rsa && chmod 644 .ssh/id_rsa.pub \
    &&ssh-keyscan github.com >> .ssh/known_hosts \
    && cp /tmp/build/gitconfig ${_HOME}/.gitconfig \
    && cp /tmp/build/download-vs-code-server.sh ${_HOME}/

RUN rm -rf /tmp/build

ENV PYTHONPATH=.:${PYTHONPATH}

CMD "zsh"



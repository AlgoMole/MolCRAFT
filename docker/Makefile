SHELL=/bin/bash
BASE_IMAGE:=chloechiu/mol_base:pt112-cu116-dev

PROJECT_NAME:=molcraft
TEAM_NAME:=bfn
PUBLIC_IMAGE:=${PROJECT_NAME}
HTTPS_PROXY:=${https_proxy}
HTTP_PROXY:=${http_proxy}
_USER := $(shell whoami)
_UID := $(shell id -u)
_GID := $(shell id -g)
PW:=rootpass

LOCAL_NAME:=exp/${_USER}_${PROJECT_NAME}
PRIVATE_IMAGE:=${LOCAL_NAME}
CONTRAINER_NAME=$(subst /,_,$(LOCAL_NAME))

PROJ_DIR:=${HOME}
SHARE_DIR:=${HOME}/share
TORCH_HOME:=${SHARE_DIR}/torch_home

CHOME:= ${HOME}/home
TO_SHARE_DIR:=${SHARE_DIR}
TO_PROJ_DIR:=${CHOME}
TO_TORCH_HOME:=${TO_SHARE_DIR}/torch_home

MAKEFILE_ := $(CURDIR)/$(firstword $(MAKEFILE_LIST))

run: _check_image _instantiate_container _exec

run_root: _check_image _instantiate_container _exec_root

run_refresh: kill run

kill:
	if docker container ls | grep -w ${CONTRAINER_NAME}; then \
		echo "container ${CONTRAINER_NAME} exists, killing it first"; \
		docker container kill  ${CONTRAINER_NAME}; fi

build_private: build_public _setup_sshkey _setup_gitconfig
	DOCKER_BUILDKIT=1 docker build --no-cache --target private \
		--build-arg PUBLIC_IMAGE=${PUBLIC_IMAGE} \
		--build-arg TORCH_HOME=${TO_TORCH_HOME} \
		--build-arg https_proxy=${HTTPS_PROXY} \
		--build-arg http_proxy=${HTTP_PROXY} \
		--build-arg _USER=${_USER}  \
		--build-arg _HOME=${CHOME}  \
		--build-arg _UID=${_UID} \
		--build-arg _GID=${_GID} \
		--build-arg PW=${PW} \
		-t ${PRIVATE_IMAGE} .

build_public:
	DOCKER_BUILDKIT=1 docker build --pull --target public --build-arg BASE_IMAGE=${BASE_IMAGE} \
	--build-arg https_proxy=${HTTPS_PROXY} \
	--build-arg http_proxy=${HTTP_PROXY} \
	-t ${PUBLIC_IMAGE} .

_exec:
	docker exec -it ${CONTRAINER_NAME} zsh

_exec_root:
	docker exec -itu 0 ${CONTRAINER_NAME} zsh

_instantiate_container: 
	touch ${HOME}/.netrc
	touch ${HOME}/.zsh_history
	if [ ! -d ${HOME}/project ]; then \
		mkdir -p ${HOME}/project;fi 
	if [ ! -d ${HOME}/vscode/vscode-server ]; then \
		mkdir -p ${HOME}/vscode/vscode-server;fi 
	if [ ! -d ${HOME}/vscode/vscode-remote-containers ]; then \
		mkdir -p ${HOME}/vscode/vscode-remote-containers; fi 
	if ! docker container ls | grep -w ${CONTRAINER_NAME}; then \
		docker run -itd --rm --shm-size=256g \
			-v ${SHARE_DIR}:${TO_SHARE_DIR} \
			-v ${HOME}/.netrc:${CHOME}/.netrc \
			-v ${HOME}/.zsh_history:${CHOME}/.zsh_history \
			-v ${PROJ_DIR}:${TO_PROJ_DIR} \
			-v ${HOME}/vscode/vscode-server:${CHOME}/.vscode-server \
			-v ${HOME}/vscode/vscode-remote-containers:${CHOME}/.vscode-remote-containers \
			--user ${_USER} \
			--name ${CONTRAINER_NAME} --gpus all $(PRIVATE_IMAGE); \
	fi; \

_check_image:
	if ! docker image ls | grep -w ${PRIVATE_IMAGE}; then \
		echo "${PRIVATE_IMAGE} image not found, try building private image first"; \
		${MAKE} -f ${MAKEFILE_} --no-print-directory build_private; \
	fi

# attach is dangerous, exit from attached shell will terminate the container
_attach:
	docker attach ${CONTRAINER_NAME}

_setup_sshkey:
	touch ../.gitignore
	if [ -z "$(cat ../.gitignore| grep ssh/)" ]; then echo "ssh/" >> ../.gitignore; fi 
	if [ -e asset/ssh/id_rsa ] && [ -e asset/ssh/id_rsa.pub ]; then \
		echo "sshkey exists, use default key";\
	else \
		if [ ! -e ~/.ssh/id_rsa ] || [ ! -e ~/.ssh/id_rsa.pub ]; then \
			echo "sshkey don't exist, try generating it"; \
			ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""; \
			echo new sshkey generated; \
		fi; \
		mkdir -p asset/ssh; \
		cp ~/.ssh/id_rsa asset/ssh/id_rsa; \
		cp ~/.ssh/id_rsa.pub asset/ssh/id_rsa.pub; \
		echo To grant access to your github account, make sure to copy the content in [~/.ssh/id_rsa.pub] to github:;\
		cat ~/.ssh/id_rsa.pub; \
	fi

_setup_gitconfig:
	touch ../.gitignore
	if [ -z "$(cat ../.gitignore| grep gitconfig)" ]; then echo "gitconfig" >> ../.gitignore; fi 
	if [ -e asset/gitconfig ]; then \
		echo "gitconfig exists, use default gitconfig"; \
	else \
		if [ ! -e ~/.gitconfig ]; then \
			echo "gitconfig don't exist, try configure it"; \
			read -p "(no space and special character allowed) Enter github username: " GIT_NAME; \
			read -p "(must be your real github account) Enter github email: " GIT_EMAIL; \
			git config --global user.name $${GIT_NAME:-${PROJECT_NAME}}; \
			git config --global user.email $${GIT_EMAIL:-${PROJECT_NAME}@gmail.com}; \
		fi; \
		cp ~/.gitconfig asset/gitconfig; \
	fi
	
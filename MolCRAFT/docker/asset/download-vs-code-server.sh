#!/bin/sh

set -e

# Auto-Get the latest commit sha via command line.
get_latest_release() {
    tag=$(curl --silent "https://api.github.com/repos/${1}/releases/latest" | # Get latest release from GitHub API
          grep '"tag_name":'                                              | # Get tag line
          sed -E 's/.*"([^"]+)".*/\1/'                                    ) # Pluck JSON value

    tag_data=$(curl --silent "https://api.github.com/repos/${1}/git/ref/tags/${tag}")

    sha=$(echo "${tag_data}"           | # Get latest release from GitHub API
          grep '"sha":'                | # Get tag line
          sed -E 's/.*"([^"]+)".*/\1/' ) # Pluck JSON value

    sha_type=$(echo "${tag_data}"           | # Get latest release from GitHub API
          grep '"type":'                    | # Get tag line
          sed -E 's/.*"([^"]+)".*/\1/'      ) # Pluck JSON value

    if [ "${sha_type}" != "commit" ]; then
        combo_sha=$(curl -s "https://api.github.com/repos/${1}/git/tags/${sha}" | # Get latest release from GitHub API
              grep '"sha":'                                                     | # Get tag line
              sed -E 's/.*"([^"]+)".*/\1/'                                      ) # Pluck JSON value

        # Remove the tag sha, leaving only the commit sha;
        # this won't work if there are ever more than 2 sha,
        # and use xargs to remove whitespace/newline.
        sha=$(echo "${combo_sha}" | sed -E "s/${sha}//" | xargs)
    fi

    printf "${sha}"
}

ARCH="x64"
U_NAME=$(uname -m)

if [ "${U_NAME}" = "aarch64" ]; then
    ARCH="arm64"
fi

archive="vscode-server-linux-${ARCH}.tar.gz"
owner='microsoft'
repo='vscode'
commit_sha=$(get_latest_release "${owner}/${repo}")

if [ -n "${commit_sha}" ]; then
    echo "will attempt to download VS Code Server version = '${commit_sha}'"

    # Download VS Code Server tarball to tmp directory.
    curl -L "https://update.code.visualstudio.com/commit:${commit_sha}/server-linux-${ARCH}/stable" -o "/tmp/${archive}"

    # Make the parent directory where the server should live.
    # NOTE: Ensure VS Code will have read/write access; namely the user running VScode or container user.
    mkdir -vp ~/.vscode-server/bin/"${commit_sha}"

    # Extract the tarball to the right location.
    tar --no-same-owner -xzv --strip-components=1 -C ~/.vscode-server/bin/"${commit_sha}" -f "/tmp/${archive}"
else
    echo "could not pre install vscode server"
fi
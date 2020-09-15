import re
import subprocess

import logging
logger = logging.getLogger("scarlet.version")

def get_public_version():
    """Get the latest publicly released version

    When there is no git repo present this must be a public release,
    so load the version from the latest sdist.

    Returns
    -------
    result: str
        The name of the latest public version.
    """
    try:
        # We can't just import scarlet._version because we are not
        # using
        f = open('_version.txt')
        _version = f.readline()
        f.close()
        return _version
    except FileNotFoundError:
        msg = ("Could not find either a git repo or pre-installed version." +
               "This should never happen, please open an issue at www.github.com/pmelchior/scarlet" +
               "so that it can be corrected.")
        raise Exception(msg)


def get_version():
    """Create the version from SCM

    If a git repo exists use the tags to create the version,
    otherwise look for a hard coded version for a release distribution
    """

    # First attempt to load git tags
    try:
        tags = subprocess.check_output(["git", "tag", "--sort=taggerdate"]).decode("utf-8").split("\n")[:-1]
    except subprocess.CalledProcessError:
        # There is no git repo found, so use the hardcoded version included with the latest distribution
        return get_public_version()

    # Find the latest tag with a public version of scarlet
    public_version = None

    while (public_version is None) and len(tags) > 0:
        tag = tags.pop()
        # Ignore scarlet tags
        if re.search("^(\d+\.)?(\d+\.)?(\*|\d+)$", tag):
            public_version = tag

    if public_version is None:
        raise Exception("Could not find a public version of scarlet in the repo")

    local_version = get_local_version(public_version)
    full_version = public_version
    if local_version is not None:
        full_version += "+" + local_version

    return full_version


def get_local_version(public_version):
    """Get the local version of scarlet
    In general this will be the latest commit, if the commit is different
    than the tagged commit. However, it is possible for forked packages to
    overwrite this function to generate their own local version.
    For example see `version.py` in github.com/lsst/scarlet.

    Returns
    -------
    local_version: str
        The local version of the package
    """
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"])[:7].decode("utf-8")
    tag_commit = subprocess.check_output(["git", "rev-list", "-n", "1", public_version])[:7].decode("utf-8")
    if tag_commit == commit:
        return None
    return "g" + commit

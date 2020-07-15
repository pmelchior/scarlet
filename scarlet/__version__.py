import subprocess

# Use the firt 7 digits of the git hash to set the version
version_root = "1.0"
try:
    __version__ = (
        version_root
        + ".dev0+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"])[:7].decode("utf-8")
    )
except:
    __version__ = version_root

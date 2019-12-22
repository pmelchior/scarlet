import os,glob

source_files = glob.glob('../scarlet/[a-z0-9]*.py', recursive=True)
sources = sorted([ ".".join(f[3:-3].split("/")) for f in source_files ])

try:
    os.mkdir('api')
except FileExistsError:
    pass

# create overview
filename = "api/scarlet.rst"
with open(filename, "w") as fp:
    fp.write("API Documentation\n")
    fp.write("=================\n\n")
    fp.write(".. toctree::\n\n")
    for source in sources:
        fp.write("    {}\n".format(source))

# create module pages
for source in sources:
    filename = "api/{}.rst".format(source)
    with open(filename, "w") as fp:
        title = "{}".format(source)
        fp.write(title + "\n")
        fp.write("=" * len(title) + "\n\n")
        fp.write(".. automodule:: {}\n".format(source))
        fp.write("    :members:\n")
        fp.write("    :undoc-members:\n")
        fp.write("    :show-inheritance:\n")

# prx-practical-lab

## Python environment creation
This repo uses the `uv` tool to manage the python virtual environment. Here are the steps to get started:
- Install `uv` using [these instructions](https://docs.astral.sh/uv/getting-started/installation/)
- Execute `uv sync` in a terminal

This will create a `.venv` folder containing the python virtual environment.

## Running python scripts
To run a python script using the virtual environment, use the command
```
uv run script.py
```

## Notebook cleaning before git-pushing
Run the following command `nbstripout --install --attributes=.gitattributes`

This will install `nbstripout` ([github](https://github.com/kynan/nbstripout)), a tool that automatically clear all outputs from notebook before committing.

## Use `git-filter-repo` to create a clean repository without solution
The following command lines will clone the current local repository and keep only the files stored in the file `keep.txt`:
``` bash
# move to the parent folder of the local prx_notebook folder
cd ..
# copy folder to a different folder named prx_notebook_students
git clone prx_notebook prx_notebook_students
cd prx_notebook_students
# filter repo keeping files defined in keep.txt
uvx git-filter-repo --paths-from-file keep.txt --force
# add remote public repo
git remote add student-repo https://github.com/navforge/prx-practical-lab.git
# push main branch to public repo
git push student-repo main
```
Note that the use of git-filter-repo is destructive, so use the original local repository (or remote) as back up.

<p align="center">
  <a href="https://github.com/destin-v">
    <img src="https://raw.githubusercontent.com/destin-v/destin-v/main/docs/pics/logo.gif" alt="drawing" width="500"/>
  </a>
</p>

# 📒 Description
<p align="center">
  <img src="docs/pics/program_logo.png" alt="drawing" width="350"/>
</p>

<p align="center">
  <a href="https://devguide.python.org/versions/">              <img alt="" src="https://img.shields.io/badge/python-^3.10-blue?logo=python&logoColor=white"></a>
  <a href="https://docs.github.com/en/actions/quickstart">      <img alt="" src="https://img.shields.io/badge/CI-github-blue?logo=github&logoColor=white"></a>
  <a href="https://black.readthedocs.io/en/stable/index.html">  <img alt="" src="https://img.shields.io/badge/code%20style-black-blue"></a>
</p>

<p align="center">
  <a href="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pre-commit.yml">  <img alt="pre-commit" src="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pre-commit.yml/badge.svg"></a>
  <a href="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pdoc.yml">        <img alt="pdoc"       src="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pdoc.yml/badge.svg"></a>
  <a href="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pytest.yml">      <img alt="pytest"     src="https://github.com/mit-ll/Apptainer-Templates/actions/workflows/pytest.yml/badge.svg"></a>
</p>

Singularity containers inherits your username and permissions from the host system.  The container allows you to install software in a controlled environment making it possible to **build once, deploy anywhere**.

To setup a container we need to: (1) create a definition file, (2) build an image, and (3) setup an overlay.  This will result in a Singularity container that has persistent storage allowing a developer to build software.

# 🐾 Guide
## User Elevation
To build a container you will need sudo privileges.

```console
sudo su -
```


## Builds
We start by building an image file from a definition file.

```console
singularity build <name>.sif <name>.def
```

## Running
To run an image use the following command:

```console
singularity shell <name>.sif
```

If you need to experiment with a temporary sandbox:

```console
singularity shell --tmp-sandbox --writable <name>.sif
```

> [!IMPORTANT]
> ...binding paths to non-existent points within the container can result in unexpected behavior when used in conjunction with the --writable flag, and is therefore disallowed. If you need to specify bind paths in combination with the --writable flag, please ensure that the appropriate bind points exist within the container. If they do not already exist, it will be necessary to modify the container and create them. [**Reference**](https://docs.sylabs.io/guides/3.5/user-guide/bind_paths_and_mounts.html#a-note-on-using-bind-with-the-writable-flag)

## Overlays
It is possible to embed an overlay image into the SIF file that holds a container. This allows the read-only container image and your modifications to it to be managed as a single file.

To add a 1 GiB writable overlay partition to an existing SIF image:

```console
singularity overlay create --size 1024 <name>.sif
singularity shell --writable <name>.sif
```

>[!NOTE]
The overlay is a separate file storage that is owned by the user who created it.  When attached to a SIF image, it is treated as a bound volume that retains its original permissions.


## Tests
The built-in tests included building (1) mamba image, (2) miniconda image, and (3) a base image.  The base image is likely your starting point for any project as it will include miniconda with Ubuntu 22.04 and the ability to add any additional Conda environments of your choosing.

```console
bash tests/build_test.sh
```

# 😎 Advance Concepts

## Dockerfile → Singularity Image
In order to convert a Dockerfile into a Singularity image file the order of operations is: (1) build a docker image from Dockerfile, (2) build a Singularity image from the Docker image.

```console
docker build -t local/<my_container>:latest .

sudo singularity build <my_container>.sif docker-daemon://local/<my_container>:latest
```

An example can be found under `tests/convert_test.sh`.  To execute the test:
```console
bash tests/convert_test.sh
```

## Activating Anaconda
There are two properties that each shell has:

* It is either a **login** or a **non-login shell**
* It is either a **interactive** or **non-interactive shell**.

A thorough explanation can be found [**here**](https://geniac.readthedocs.io/en/latest/conda.html).

When activating an Anaconda environment you are normally in a login and interactive terminal.  When you are working in the definition file of Singularity (i.e. `%post`), you are in a non-login terminal.  This means that some processes the normally happen for a login terminal do not get applied when working with Singularity.

The way to fix this is by applying the log properties to your Singularity container via:

```console
conda create -n myenv python=3.10   # create a custom environment

. /opt/miniconda3/etc/profile.d/conda.sh   # activate login properties to terminal

conda activate myenv    # activating your custom environment
```

## Running Singularity on SLURM
Instructions for setting up Singularity to run on SLURM are found [**here**](https://nfdi4ing.pages.rwth-aachen.de/knowledge-base/how-tos/all_articles/how_to_run_a_job_on_the_cluster_using_slurm_and_singularity/).  In general, each Singularity image needs to be able to execute on its own.  The SLURM scheduler will dispatch the Singularity images to each node and proceed to run.



## Manually Building Writable Partitions
You can use tools like `dd` and `mkfs.ext3` to create and format an empty `ext3` file system image, which holds all changes made in your container within a single file. Using an overlay image file makes it easy to transport your modifications as a single additional file alongside the original `SIF` container image.

This script will create an overlay with 1 GB of storage which can be attached to a Singularity
image.

```console
dd if=/dev/zero of=overlay.img bs=1M count=1000 && \
    mkfs.ext3 -d overlay overlay.img
```

Then attach the overlay with your image to create a writable container.

```console
sudo singularity shell --overlay overlay.img <name>.sif
```

## Persistent Overlays
A persistent overlay is a directory or file system image that “sits on top” of your immutable SIF container. When you install new software or create and modify files the overlay will store the changes.

If you want to use a SIF container as though it were writable, you can create a directory, an ext3 file system image, or embed an ext3 file system image in SIF to use as a persistent overlay. Then you can specify that you want to use the directory or image as an overlay at runtime with the `--overlay` option, or `--writable` if you want to use the overlay embedded in SIF.

If you want to make changes to the image, but do not want them to persist, use the `--writable-tmpfs` option. This stores all changes in an in-memory temporary file system which is discarded as soon as the container finishes executing.

You can use persistent overlays with the following commands:

```console
run
exec
shell
instance.start
```

Here is an example of applying a `.sif` overlay on top of an existing `img`.

```console
sudo singularity shell --overlay overlay.img <dev>.sif
```

# ♖ Distribution

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

# What is pycrazyswarm?

Pycrazyswarm is a series of RL environments, trained using Proximal Policy Optimizations (PPOs), complete with a simulator for CrazyFlie drones, and a 3D visualizer. It's compatible with Tensorboard, and a host of other tools. We train using [https://github.com/hill-a/stable-baselines](Stable Baselines)'s implementation of PPO, and the simulator is loosely based off of the one found in [https://github.com/USC-ACTLab/crazyswarm](Crazyswarm).

# Installation

On a fresh Ubuntu box:
`$ sudo apt install git make gcc swig libpython-dev python-numpy python-yaml python-matplotlib
$ git clone https://github.com/arya-k/pycrazyswarm.git
$ cd pycrazyswarm/crazyenv/cfsim
$ chmod +x build.sh && ./build.sh`

Then, create a virtualenv, and install all the packages in `reqs.txt`.

From there, run any of the scripts in the `/scripts` directory by moving them to the base of the project, and running them within your virtualenv.

# Project website

More information about the project can be found on the project website: [https://swarm.sites.tjhsst.edu](https://swarm.sites.tjhsst.edu).
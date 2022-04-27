# MitsubaPyOptiX
This is a custom python, OptiX based path tracing engine that renders Mitsuba formatted scenes.
This renderer was used for the paper 
["Fast and Lightweight Path Guiding Algorithm on GPU"](https://diglib.eg.org/handle/10.2312/pg20211379) 
by Juhyeon Kim and Young Min Kim (Pacific Graphics 2021 short paper).

(20220428) currently being updated


## vs Mitsuba
MitsubaPyOptiX is python based, so is much easier than C++!
Also, it exploits OptiX with megakernel architecture, 
which is faster than wavefront architecture used in Mitsuba2.

| Name           | Backend                        | Language         |
|----------------|--------------------------------|------------------|
| Mitsuba        | CPU based                      | C++              |
| Mitsuba2       | OptiX, wavefront architecture  | C++              |
| MitsubaPyOptiX | OptiX, megakernel architecture | Python           |

## Prerequisites
You need OptiX 6.5 which could be downloaded 
[here](https://developer.nvidia.com/designworks/optix/downloads/legacy).
(7.0 is not supported currently)
Do not forget to set `OptiX_INSTALL_DIR` and `LD_LIBRARY_PATH` in bash file.
Then create an environment and install requirements.
```
conda create --name pyoptixpathtracer python=3.8
conda activate pyoptixpathtracer
pip install -r requirements.txt
```

Also, Install custom PyOptiX that slightly modified the [original one](https://github.com/MathGaron/PyOptiX).
```
git clone https://github.com/juhyeonkim95/PyOptiX.git
cd PyOptiX
python setup.py install
```

## Scene Data
The scene data could be downloaded [this site](https://benedikt-bitterli.me/resources/).

## Usage
To render the scene use following.
Example config json files are uploaded.
```
cd src
python main.py ../configs_example/brdf.json
```

Details for configuration could be found [here](configs_example/README.md).

## Citation
If you find this useful for your research, please consider to cite:
```
@inproceedings {10.2312:pg.20211379,
booktitle = {Pacific Graphics Short Papers, Posters, and Work-in-Progress Papers},
editor = {Lee, Sung-Hee and Zollmann, Stefanie and Okabe, Makoto and Wünsche, Burkhard},
title = {{Fast and Lightweight Path Guiding Algorithm on GPU}},
author = {Kim, Juhyeon and Kim, Young Min},
year = {2021},
publisher = {The Eurographics Association},
ISBN = {978-3-03868-162-5},
DOI = {10.2312/pg.20211379}
}
```
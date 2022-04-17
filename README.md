# MitsubaPyOptiX

This is a custom python, OptiX based path tracing engine that renders Mitsuba formatted scenes.
(To be updated)

## vs Mitsuba
MitsubaPyOptiX is python based, so is much easier than C++!
Also, it exploits OptiX with megakernel architecture, 
which is faster than wavefront architecture used in Mitsuba2.

| Name           | Rendering Method               | Code             |
|----------------|--------------------------------|------------------|
| Mitsuba        | CPU based                      | C++              |
| Mitsuba2       | OptiX, wavefront architecture  | C++              |
| MitsubaPyOptiX | OptiX, megakernel architecture | Python, some C++ |

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

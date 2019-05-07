# Particle tracker

Needs opencv with opencv-contrib
Install conda environment from envs folder

After instaling the conda environment with all the needed packages execute any of the following scripts:

```python
python multiple_track video_path
```


```python
python tracker2 video_path
```

The ping-pong.py file does the same on a youtube video from the classic ping-pog game.

docs compilation done by executing following command in the docs folder:

```python
make html
```
Aim of the module is to detec and track fluorescent particles in micro-bioreactors.
With the position of the particles and a proper pixel calibration, the velicity of the aprticles is estimated and stored for further fluid analysis.

![img1](/images/spiral.png)

![img2](/images/multiple_particles.png)

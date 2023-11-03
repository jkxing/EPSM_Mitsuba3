# EPSM
Siggraph Asia 2023 Paper "Extended Path Space Manifold for Physically Based Differentiable Rendering"

# Install
## Set Environment
```bash
conda create -n EPSM python=3.9 #create python env
conda activate EPSM
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  #using pytorch cuda 11.8
pip install geomloss[full] # geomloss for matching ,linux only, make sure it passes the tests
pip install opencv-python tensorboardX numpy==1.23 h5py scikit-learn tensorboard tqdm # install other dependency
pip install smplpytorch chumpy #for human scene
mkdir build && cd build && cmake .. -GNinja && ninja && source setpath.sh #build mitsuba and set python path
```
### some tips
if this error occurs
```bash
ImportError: /home/xxx/miniconda3/envs/EPSM_Mitsuba/lib/python3.9/site-packages/torch/lib/../../../.././libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xxx/EPSM_Mitsuba3/build/libdrjit-core.so)
```
try:
```bash
cd ~/miniconda3/envs/EPSM_Mitsuba/lib && sudo ln -S libstdc++.so.6  /usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
## Run
Download `data/` and `result_sample/` (optional) from here(todo) and extract it into `EPSM/` folder ([Google Drive](https://drive.google.com/drive/folders/14Rm27_l5nLsCJ--b3jxxvS6i8_T-NBTz?usp=share_link))
```bash
cd EPSM
python optim.py METHOD EXP_NAME 
# eg. python optim.py manifold bathroom
```
METHOD: manifold/manifold_caustic/manifold_hybrid/manifold_caustic_hybrid

EXP_NAME: follow filenames in `EPSM/exp/`


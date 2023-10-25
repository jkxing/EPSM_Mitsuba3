# EPSM
Siggraph Asia 2023 Paper "Extended Path Space Manifold for Physically Based Differentiable Rendering"

# Install
## Set Environment
```bash
conda create -n EPSM python=3.9 #create python env
conda activate EPSM
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  #using pytorch cuda 11.8
pip install geomloss[full] # geomloss for matching ,linux only, make sure it passes the tests
pip install opencv-python tensorboardX numpy h5py scikit-learn # install other dependency
mkdir build && cd build && cmake .. -GNinja && ninja && source setpath.sh #build mitsuba and set python path
```
## Run
Download `data/` and `result_sample/` (optional) from here(todo) and extract it into `EPSM/` folder
```bash
cd EPSM
python optim.py METHOD EXP_NAME 
# eg. python optim.py manifold bathroom
```
METHOD: manifold/manifold_caustic/manifold_hybrid/manifold_caustic_hybrid

EXP_NAME: follow filenames in `EPSM/exp/`


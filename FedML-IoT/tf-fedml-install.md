## Pip3 install (python 3.7 from /usr/bin/)
- Can't install within conda env due to incompatible tensorflow version

pip install --upgrade pip
pip3 install --upgrade setuptools
pip3 install numpy
sudo apt install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
pip3 install keras_applications --no-deps
pip3 install keras_preprocessing --no-deps
pip3 install h5py
pip3 install pybind11
pip3 install -U --user six wheel mock

pip3 uninstall tensorflow
pip3 install tensorflow-2.3.0-cp37-none-linux_armv7l.whl

pip3 install mpi4py openpyxl


### Note: reinstall compatible versions of pandas and numpy with --no-binary flag as needed

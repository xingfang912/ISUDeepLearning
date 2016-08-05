# It's a good idea to run the general software updater before beginning.
# Open a terminal Ctrl-Alt-T
# Note that this was created for ubuntu 14.04

# INSTALL CUDA TOOLKIT 7.5
# cuda toolkit 7.5 .deb for ubuntu 14.04 is in this folder.
# Open the terminal and navigate to this directory.
cd ~/Documents/install_instructions/ # might be different on your system.
# If you already have the cuda .deb package (~2GB), then skip step 1.

# STEP 1
# add repository as a source and download cuda toolkit .deb package file. skip if you already have the .deb
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get --purge remove "nvidia*"
sudo apt-get --purge remove "cuda*"
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb

# STEP 2
# add cuda deb package to repository.
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
# update repository
sudo apt-get update
# sudo apt-get upgrade
# install the cuda deb.
sudo apt-get install cuda
# set environment variables in .bashrc
echo "export PATH=/usr/local/cuda-7.5/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
# if nvidia-352 breaks your desktop env, then reinstall without libcheese*
# sudo apt-get install nvidia-current
# sudo reboot
# sudo apt-get --purge remove "libcheese*"
# sudo apt-get install cuda
# sudo apt-get install ubuntu-desktop
# sudo reboot

# STEP 3 (Optional)
# INSTALL CUDNN V3.0 : if these files are available to you.
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/

# STEP 4
# INSTALL THEANO
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ git libatlas3gf-base libatlas-dev
sudo apt-get install python-matplotlib
sudo pip install Theano
# Fix shared object file problem.
sudo ldconfig /usr/local/cuda-7.5/lib64
# Make sure theano.rc is copied into home directory. This hidden file is included in this directory.
# you may want to modify theano.rc if you have multiple gpus, or no gpus.
# to find out the names of your gpus do the following:
./usr/local/cuda-7.5/samples/1_Utilities/deviceQuery/deviceQuery

# STEP 5 (Optional)
# INSTALL CAFFE
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libatlas-base-dev # Not sure if this is needed.
sudo update-alternatives --config libblas.so # selection option 0. # Not sure if needed.
cp -r ./caffe-master ~/caffe
cd ~/caffe
cp Makefile.config.example Makefile.config
# I uncommented the USE_CUDNN := 1 line since I have a gpu.
make all # -j20 # this can be used to multicore compile. 20 means use 20 threads.
make test
make runtest
make pycaffe
echo "export PYTHONPATH=\$CAFFE_HOME/python:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
sudo pip install -U scikit-image # might need to do this in a new terminal window.
sudo pip install protobuf
# Now you should be able to use caffe in python (import caffe)

# STEP 6
# INSTALL ROS Jade: Many dependencies come with this, including opencv
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net:80 --recv-key 0xB01FA116
sudo apt-get update
sudo apt-get install ros-jade-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/jade/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python-rosinstall
# --------------------------------------
# Initializing your workspace: (Optional)
mkdir -p ~/Documents/catkin_ws/src
cd ~/Documents/catkin_ws/src
catkin_init_workspace
cd ~/Documents/catkin_ws/
catkin_make
echo "source ~/Documents/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/Documents/catkin_ws/src
#catkin_create_pkg mirage std_msgs rospy roscpp
#cd ~/Documents/catkin_ws/
#catkin_make
#echo "source ~/Documents/catkin_ws/src/mirage/scripts/setup.bash" >> ~/.bashrc
# -------------------------------------
sudo apt-get install python-pip
sudo pip install pyudev

# STEP 7 (Optional)
# INSTALL GAZEBO SIMULATOR
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install gazebo7
# For developers that work on top of Gazebo, one extra package
sudo apt-get install libgazebo7-dev

# STEP 8
# INSTALL ISUDeepLearning
# Download the source from github: https://github.com/xingfang912/ISUDeepLearning.git
# TODO Show details of how to clone the source using command line.
#   If you want to be lazy, you can just download the zip file and unpack it.
# Navigate into the ISUDeepLearning directory and run setup.py as follows:
sudo python setup.py develop
# this will install the package so it can be imported in other python programs.

# HOW TO UPLOAD CHANGES TO GITHUB AND PYPI
# If you modify the code, you can upload the changes to github by uploading the modified files
# so that they overwrite the existing versions.
# to push your updates to the pypi repository (so pip can be used to install), use
sudo python setup.py register sdist upload
# Note that you'll need a pypi account in order to do this.


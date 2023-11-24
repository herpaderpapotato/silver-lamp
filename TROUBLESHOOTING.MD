Using this on a fresh WSL install based on windows 10 and wsl --install Ubuntu

I had to do the following to get miniconda going https://docs.conda.io/projects/miniconda/en/latest/index.html

    cd ~
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
and then

    ~/miniconda3/bin/conda init bash
and then exit wsl, and run it again


I had to do the following for opencv to work.

    sudo apt-get update
    sudo apt-get install ffmpeg libsm6 libxext6


For the cuda predictions to work, I needed to do the following:

    sudo apt-key del 7fa2af80
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu-keyring.gpg
    sudo mv cuda-wsl-ubuntu-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-get update
    sudo apt-get install cuda-toolkit


I had to do the following to set up the conda environment and run the code

    conda create -n silverlampenv python=3.10
    conda activate silverlampenv
    cd ~
    git clone https://github.com/herpaderpapotato/silver-lamp
    cd silver-lamp
    pip install -r requirements.txt
    python inference.py --video "video path here"
or put a mp4 in the silver-lamp folder and just run 
    python inference.py
Usage instructions on the [README.MD](README.MD)


To copy files in and out of the Ubuntu instance I had to open folder in explorer \\wsl.localhost\Ubuntu\home, and then navigate to my username and the silver-lamp folder.

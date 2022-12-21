path="./preprocessing/data/models/"
mkdir -p $path
cd $path

# Download pre-trained model of sdps-net
for model in "LCNet_CVPR2019.pth.tar" "NENet_CVPR2019.pth.tar"; do
    wget http://www.visionlab.cs.hku.hk/data/SDPS-Net/models/${model}
done

cd ../../../

# Download pre-trained model
wget http://www.visionlab.cs.hku.hk/data/s3nerf/data.tgz
tar -xzvf data.tgz
rm data.tgz

# Download dataset (including both synthetic and captured datasets)
wget http://www.visionlab.cs.hku.hk/data/s3nerf/dataset.tgz
tar -xzvf dataset.tgz
rm dataset.tgz

# Download envmap
wget http://www.visionlab.cs.hku.hk/data/s3nerf/envmap.tgz
tar -xzvf envmap.tgz
rm envmap.tgz
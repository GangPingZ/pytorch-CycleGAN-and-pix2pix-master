set -ex
python train.py --dataroot /home/data/zgp/dataset/EUVP/test_samples-20230425T075315Z-001/test_samples \
                --name EUVP_cyclegan_A22B    \
                --model cycle_gan

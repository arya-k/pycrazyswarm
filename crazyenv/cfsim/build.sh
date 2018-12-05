#!/usr/bin/env bash
touch __init__.py

git clone https://github.com/USC-ACTLab/crazyflie-firmware.git
cd crazyflie-firmware
git checkout 0858430bab5774e35f0412c2a1cf7705fd7b1b25

cd ..
make

rm -Rf crazyflie-firmware
echo 'Firmware successfully built.'

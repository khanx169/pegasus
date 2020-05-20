#!/bin/bash
wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz -P data/
cd data/
tar -xf cifar10.tgz
rm cifar10.tgz

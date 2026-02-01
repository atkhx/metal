#!/bin/bash

# http://yann.lecun.com/exdb/mnist/

curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --fail --output train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --fail --output train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --fail --output t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --fail --output t10k-labels-idx1-ubyte.gz

gunzip ./*.gz
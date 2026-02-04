#!/bin/bash

echo "1. Download original dataset archive"

if [ -f "./cifar100.tar.gz" ]; then
  echo "file cifar100.tar.gz already exists";
else
  curl http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz --fail --output ./cifar100.tar.gz || exit 1
fi;

echo "2. Extract archive"
tar xzvf ./cifar100.tar.gz

echo "3. Move train batch to cifar100-train-data.bin"
mv ./cifar-100-binary/train.bin ./cifar100-train-data.bin

echo "4. Move test batch to cifar100-test-data.bin"
mv ./cifar-100-binary/test.bin ./cifar100-test-data.bin

echo "5. Remove extracted data"
rm -rf ./cifar-100-binary

#!/bin/bash

echo "1. Download original dataset archive"

if [ -f "./cifar10.tar.gz" ]; then
  echo "file cifar10.tar.gz already exists";
else
  curl http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz --fail --output ./cifar10.tar.gz || exit 1
fi;

echo "2. Extract archive"
tar xzvf ./cifar10.tar.gz

echo "3. Concatenate train batches into cifar10-train-data.bin"
rm ./cifar10-train-data.bin
for i in {1..5} ; do
    cat "./cifar-10-batches-bin/data_batch_$i.bin" >> ./cifar10-train-data.bin
done

echo "4. Move test batch to cifar10-test-data.bin"
mv ./cifar-10-batches-bin/test_batch.bin ./cifar10-test-data.bin

echo "5. Remove extracted data"
rm -rf ./cifar-10-batches-bin

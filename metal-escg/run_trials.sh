#!/bin/bash
for i in {1..10}
do
    ./build/metal $i &
done
wait


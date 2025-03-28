#!/bin/bash
for i in {1..10}
do
    ./escg $i &
done
wait


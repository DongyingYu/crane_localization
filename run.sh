#!/usr/bin/bash

./build/test_websocket

cat ./conf/crane_info.txt
do
   ./build/test/test_system ${line} ./conf/pipeline.yaml 0 
done




#! /bin/bash
# ./build/test/test_websocket
# head -1 means Only read the first row data.
cat ./conf/crane_info.txt | head -1 | while read line;
do
   ./build/test/test_system ${line} ./conf/pipeline_online.yaml 0 & 
done
wait

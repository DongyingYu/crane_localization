#! /bin/bash
# head -1 仅读取第一行数据
#count=0;
#cat ./conf/crane_info.txt | head -2 |tail -n +2 | while read line;
#do
#   let count+=1;
#   ./build/test/test_system ${line} ./conf/pipeline_online.yaml 6000
# > runinfo$count.log &
# ./build/test/test_websocket
# head -n 1 means Only read the first row data. tail -n +1 means read from the first line;
count=0;
cat ./conf/crane_info.txt | head -n 2 | tail -n +2 | while read line;
do
   let count+=1;
   ./build/test/test_system ${line} ./conf/pipeline_online.yaml 0 > log/run_info$count.log & 
done

#! /bin/bash

./SequenceGuesserData.sh

cat trainingdata.json | \
   tr '\n' ' ' | \
   tr '\[' '{' | \
   tr '\]' '}' | \
   sed 's/[	 ]//g' | \
   sed 's/^.//;s/.$//' | \
   sed 's/},/},@/g' | 
   tr '@' '\n'

echo ''

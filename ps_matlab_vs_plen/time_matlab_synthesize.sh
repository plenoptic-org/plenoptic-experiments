#!/usr/bin/env bash

start=$(date '+%s.%N')
matlab -nodesktop -nodisplay -r "synthesize('$1', $2, '$3'); exit;"
stop=$(date '+%s.%N')
elapsed=$(bc -l <<< "$stop - $start")
echo $elapsed

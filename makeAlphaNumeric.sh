#!/bin/bash

for x in *.xst; do
if [ ${x:9:1} = "a" ]; then
	mv $x "${x:0:9}b.xst"
	mv ${x:0:9}.xst $x
fi
done

#!/bin/bash
ID=$1
ID2=$2
i=$3
base=/tigress/changgoo/ARM/
base=/tigress/changgoo/ARM/
EXE=../pyathena/refine_rst.py
num=$(printf "%04d" "$i")
echo python $EXE -f $base/$ID/$ID.$num.rst -d $base/$ID/rst_split/ -i $ID2 --split

#!/bin/bash

Compiler=arm-linux-androideabi-clang++
$Compiler -static-libstdc++ ./parse_perf.cpp -o ./parse_perf

adb push ./parse_perf /data/local/Working_dir/
echo "pushed governor to board, now chmodding"
adb shell chmod +x /data/local/Working_dir/"$1"
echo "done"
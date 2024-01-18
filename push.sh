#!/bin/bash

# adb push governor /data/local/Working_dir/
# echo "pushed governor to board, now chmodding"
# adb shell chmod +x /data/local/Working_dir/governor
# echo "done"
# echo "governor now has execute rights, now running"
adb shell "cd /data/local/Working_dir/ && ./governor"

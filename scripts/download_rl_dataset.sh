#!/bin/bash

DIR="rl_data"

if [ ! -d "DIR" ]; then
  echo "mkdir"
  mkdir -p "$DIR"
fi

cd "$DIR"
wget http://142.171.233.124/train.pkl
wget http://142.171.233.124/test.pkl

echo "Download finished!!"
#!/bin/bash

DIR="rl_data"

if [ ! -d "DIR" ]; then
  echo "mkdir"
  mkdir -p "$DIR"
fi

# shellcheck disable=SC2164
cd "$DIR"
wget http://142.171.233.124/train.pkl
wget http://142.171.233.124/valid.pkl

echo "Download finished!!"
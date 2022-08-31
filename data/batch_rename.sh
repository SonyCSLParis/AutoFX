#!/usr/bin/env bash

for file in *.wav; do
  arrFile=(${file//-/ });
  mv "${file}" "${arrFile[3]}";
  done



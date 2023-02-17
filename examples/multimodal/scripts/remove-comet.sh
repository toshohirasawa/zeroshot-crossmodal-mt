#!/bin/bash -eu

FILE=$1

sed -i "/COMET/d" ${FILE}

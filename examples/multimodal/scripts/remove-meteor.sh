#!/bin/bash -eu

FILE=$1

# STATE=$(grep "METEOR = 0" ${FILE} | wc -l)
# if [ $STATE -eq 1 ]; then
#     echo ${FILE}
# fi

sed "/METEOR/d" ${FILE} >${FILE}.bak
mv ${FILE}.bak ${FILE}

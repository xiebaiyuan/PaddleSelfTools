#!/usr/bin/env bash

# set -o xtrace

extension=$1

convert () {
    perl -pi -e "s/lite::/lite_${1}::/g" "${2}"
    perl -pi -e "s/namespace lite /namespace lite_${1} /g" "${2}"
}

revert () {
    perl -pi -e "s/lite_[\w]*::/lite::/g" "${2}"
    perl -pi -e "s/namespace lite_[\w]*/namespace lite/g" "${2}"
}

if [[ $2 == "revert" ]]; then
    for file in $(find lite -name "*\.*")
    do
        echo "reverting ${file}"
        revert $extension $file
    done
#    for file in $(find test -name "*\.*")
#    do
#        echo "reverting ${file}"
#        revert $extension $file
#    done
else
    for file in $(find lite -name "*\.*")
    do
        echo "converting ${file}"
        convert $extension $file
    done
#    for file in $(find test -name "*\.*")
#    do
#        echo "converting ${file}"
#        convert $extension $file
#    done
fi

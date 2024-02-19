#!/bin/bash
SCENES=(bonfire bunny cat colorful cornell diorama green key lowDepth macro simpleSetting street timelapse buildings cars class colorless diffuse face highFrequency largeDepth lowFrequency reflective singleObject text volumetric)
ASPECTS=(2.276 1.8327 1.9157 1.9058 1.783 1.937 1.9175 1.909 2.122 1.9846 1.8266 1.816 2.046 1.8323 2.003 2.3807 1.8084 2.02762 1.885 1.975 2.0213 2.0223 2.02762 1.873 1.8658 1.89395)
RANGES=(0.015_0.075 0.0_0.15875 0.0075_0.1 0.04625_0.1 0.055_0.0975 0.0825_0.2125 0.045_0.10375 0.0475_0.1325 0.135_0.1575 0.025_0.1625 0.1075_0.1525 0.04_0.0725 0.0_0.0725 0.0125_0.105 0.0375_0.1575 0.0225_0.145 0.04_0.1025 0.0_0.13 0.0_0.11 0.0_0.1125 0.0_0.20875 0.0_0.115 0.0_0.13 0.0625_0.08775 0.0125_0.1075 0.0_0.1225)
RUN_TEST_SCRIPT=./runTest.sh
DATA_PATH=../build/data
REF_PATH=../build/references
OUT_PATH=../build/out
REPORT_FILE=./report.csv
> $REPORT_FILE

for i in {0..25}; do
    QUALITY=$($RUN_TEST_SCRIPT $DATA_PATH/${SCENES[$i]} $REF_PATH/${SCENES[$i]}.exr ${ASPECTS[$i]} ${RANGES[$i]})
    RESULT_PATH=$OUT_PATH/${SCENES[$i]}
    mkdir -p $RESULT_PATH
    mv $OUT_PATH/*.exr $RESULT_PATH/
    QUALITY=$(echo "$QUALITY" | tail -n1)
    echo ${SCENES[$i]} $QUALITY >> $REPORT_FILE
done

# This script extracts the dataset archive and prepares the data into the RGBA 32float EXR files with input data and reference for measurements
# The only parameter is the output directory
# The link below can be changed to any of the HDR datasets on: https://www.fit.vutbr.cz/~ichlubna/lf
TEMP=$(mktemp -d)
wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/hdrData/cornell.zip -P $TEMP
#cp cornell.zip $TEMP
unzip $(ls $TEMP | head -1) -d $TEMP
OUT=$1/data/
mkdir $OUT
NAMES=($OUT"00_00-topLeft.exr" $OUT"00_01-topRight.exr" $1"0.5_0.5-centerReference.exr" $OUT"01_00-bottomLeft.exr" $OUT"01_01-bottomRight.exr")
ID=0
for filename in $TEMP/*.exr; do
    f="$(basename -- $filename)"
    ffmpeg -y -i $TEMP/$f -pix_fmt gbrapf32le ${NAMES[$ID]}
    ID=$(( $ID+1 ))
done
rm -rf $TEMP

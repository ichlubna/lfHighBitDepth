# This script extracts the dataset archive and prepares the data into the RGBA 32float EXR files with input data and reference for measurements
# The only parameter is the output directory
# The link below can be changed to any of the HDR datasets on: https://www.fit.vutbr.cz/~ichlubna/lf
TEMP=$(mktemp -d)
wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/hdrData/cornell.zip -P $TEMP
#cp cornell.zip $TEMP
unzip $(ls $TEMP | head -1) -d $TEMP
mkdir $1/data
for filename in $TEMP/*.exr; do
    f="$(basename -- $filename)"
    ffmpeg -i $TEMP/$f -pix_fmt gbrapf32le $1/data/$f
done
mv $1/data/$(ls $1/data | head -3 | tail -n -1) $1/0.5_0.5-centerReference.exr
mv $1/data/$(ls $1/data | head -1 | tail -n -1) $1/data/00_00-topLeft.exr
mv $1/data/$(ls $1/data | head -2 | tail -n -1) $1/data/00_01-topRight.exr
mv $1/data/$(ls $1/data | head -4 | tail -n -1) $1/data/01_00-bottomLeft.exr
mv $1/data/$(ls $1/data | head -5 | tail -n -1) $1/data/01_01-bottomRight.exr
rm -rf $TEMP

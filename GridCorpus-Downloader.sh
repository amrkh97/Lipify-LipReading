mkdir "gridcorpus"
cd "gridcorpus" || exit
mkdir "raw" "align" "video"
cd "raw" && mkdir "align" "video"

for i in `seq $1 $2`
do
    printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
    
    cd "align" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" && cd ..
    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..
    unzip -q "video/s$i.zip" -d "../video"
    tar -xf "align/s$i.tar" -C "../align"
    
done

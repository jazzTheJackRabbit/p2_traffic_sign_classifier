virtualenv -p python3 ./
./bin/python install -r require.txt

wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
mkdir data
unzip traffic-signs-data -d data

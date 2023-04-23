rm -rf MetBrewer
git clone https://github.com/BlakeRMills/MetBrewer
cd MetBrewer/Python/ || exit
python3 setup.py install
cd ../../ || exit
rm -rf MetBrewer/
rm -rf met_brewer.egg-info
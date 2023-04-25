mkdir data && cd data
mkdir mug_grasp && cd mug_grasp

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GI33ZHKMB5yuZKDMkNhuUxOlEyrR68mR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GI33ZHKMB5yuZKDMkNhuUxOlEyrR68mR" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yhRegjfRfzyNMEmOFxU2p2ZRdO57pAdy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yhRegjfRfzyNMEmOFxU2p2ZRdO57pAdy" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

cd .. && mkdir mug_place && cd mug_place
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GcA5Owj6djlsVarOdOfKfkeFBiQIOEdS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GcA5Owj6djlsVarOdOfKfkeFBiQIOEdS" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xtxZygEzTpfmjlJKAA6O4l2uAA9s6dnQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xtxZygEzTpfmjlJKAA6O4l2uAA9s6dnQ" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

# test grasp
# https://drive.google.com/file/d/1GI33ZHKMB5yuZKDMkNhuUxOlEyrR68mR/view?usp=share_link

# train grasp
# https://drive.google.com/file/d/1yhRegjfRfzyNMEmOFxU2p2ZRdO57pAdy/view?usp=sharing

# test place
# https://drive.google.com/file/d/1GcA5Owj6djlsVarOdOfKfkeFBiQIOEdS/view?usp=sharing

# train place
# https://drive.google.com/file/d/1xtxZygEzTpfmjlJKAA6O4l2uAA9s6dnQ/view?usp=share_link

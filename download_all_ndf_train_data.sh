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

cd .. && mkdir bowl_grasp && cd bowl_grasp
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15E5boAx3eefkuNQVkOv-X-yzoQgzH-ov' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15E5boAx3eefkuNQVkOv-X-yzoQgzH-ov" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1capE6VBNZu5yKjL525mMYuhVUqnLmT9S' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1capE6VBNZu5yKjL525mMYuhVUqnLmT9S" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

cd .. && mkdir bowl_place && cd bowl_place
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Djx3DZKccF6oBBcKl1Jezs2LNDEzegMG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Djx3DZKccF6oBBcKl1Jezs2LNDEzegMG" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KxAsV33uOMsXgFCCTxcvPq0DkpU5S_z7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KxAsV33uOMsXgFCCTxcvPq0DkpU5S_z7" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

cd .. && mkdir bottle_grasp && cd bottle_grasp
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUC07TynJAT1BmFU11k9yo8NfauOBDm1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bUC07TynJAT1BmFU11k9yo8NfauOBDm1" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kGQ_vEyl42OGVOe38JoNmTL0E5VevCL5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kGQ_vEyl42OGVOe38JoNmTL0E5VevCL5" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

cd .. && mkdir bottle_place && cd bottle_place
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V8VmZ4gzCX2ub22wYfOJTlqSEESBW1LA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1V8VmZ4gzCX2ub22wYfOJTlqSEESBW1LA" -O test_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WGs6znQRRT8rAzdkYC--pGFxgv257cwG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WGs6znQRRT8rAzdkYC--pGFxgv257cwG" -O train_data.zip && rm -rf /tmp/cookies.txt
unzip train_data.zip && rm -rf train_data.zip
unzip test_data.zip && rm -rf test_data.zip

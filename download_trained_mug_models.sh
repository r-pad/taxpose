cd trained_models
mkdir ndf && cd ndf 
mkdir mug && cd mug
mkdir arbitrary && cd arbitrary
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vN_0Vh1RC7jUnOzTa0IQs0_TC8GQslOu' -O grasp.ckpt
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ohtuv6Fj4vwYVNWhoO-_5bUI1dYJYjZQ' -O place.ckpt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vN_0Vh1RC7jUnOzTa0IQs0_TC8GQslOu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vN_0Vh1RC7jUnOzTa0IQs0_TC8GQslOu" -O grasp.ckpt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ohtuv6Fj4vwYVNWhoO-_5bUI1dYJYjZQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ohtuv6Fj4vwYVNWhoO-_5bUI1dYJYjZQ" -O place.ckpt && rm -rf /tmp/cookies.txt
cd .. && mkdir upright && cd upright
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16kak-38a0BaqP2k_vwZ0fxexFSJsD5zN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16kak-38a0BaqP2k_vwZ0fxexFSJsD5zN" -O grasp.ckpt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Zea5jDtHwTxeTTXynVhTeXtsoYe9j2V-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Zea5jDtHwTxeTTXynVhTeXtsoYe9j2V-" -O place.ckpt && rm -rf /tmp/cookies.txt 
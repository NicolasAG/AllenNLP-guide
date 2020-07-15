#!/usr/bin/env bash

# get organization / user / account name
export ORG_NAME=$(eai organization get --fields name --no-header)
export USER_NAME=$(eai user get --fields name --no-header)
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
#export ACCOUNT_ID=$(eai account get --fields id --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

echo "account id: $ACCOUNT_ID"

echo "pushing all files to code_allennlp_guide ..."
# ignore models, .git, __pycache__, tmp, logs, current and parent folders
all_files=$(ls -I . -I ..)
for f in $all_files
do
  eai data push "code_allennlp_guide" $f:$f
done
echo "done. now submitting job..."


eai job submit \
    --image registry.console.elementai.com/$ACCOUNT_ID/allennlp \
    --data $ORG_NAME.$ACCOUNT_NAME.data_clutrr1:/clutrr \
    --data $ORG_NAME.$ACCOUNT_NAME.code_allennlp_guide:/allennlp \
    --cpu 1 \
    --mem 8 \
    --gpu 1 \
    --gpu-mem 6 \
    -- bash -c "python -m spacy download en_core_web_sm && cd /allennlp && chmod +x *.sh && ./allen_scripts.sh"
    #-- bash -c "while true; do sleep 60; done;"
    #-- bash -c "python -m spacy download en_core_web_sm && cd /allennlp && rm -rf models/* && cd src && python train.py"
#eai job exec --last -- bash
eai job log -f --last



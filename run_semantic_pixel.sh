#/bin/sh
IMG="data/frankfurt_val.png"
cd ops/lbp_semantic_pw_pixel &&
python setup.py install &&
cd ../../ &&

python main_semantic.py --img=$IMG --checkpoint-semantic data/params/semantic/pixel_model.cpt --checkpoint-esp-net dependencies/ESPNet/pretrained --pairwise-type pixel 
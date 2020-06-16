#/bin/sh
IMG="data/frankfurt_val.png"
cd ops/lbp_semantic_pw &&
python setup.py install &&
cd ../../ &&
python main_semantic.py --img=$IMG --checkpoint-semantic data/params/semantic/global_model.cpt --checkpoint-esp-net dependencies/ESPNet/pretrained --pairwise-type global --with-edges 
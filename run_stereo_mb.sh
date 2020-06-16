#/bin/sh
cd ops/lbp_stereo
python setup.py install
cd ../../

python main_stereo.py --model bp+ms+h --checkpoint-unary "data/params/stereo/MB/unary_best.cpt" --checkpoint-matching "data/params/stereo/MB/matching_lvl0_best.cpt" "data/params/stereo/MB/matching_lvl1_best.cpt" "data/params/stereo/MB/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/stereo/MB/affinity_best.cpt" --checkpoint-crf "data/params/stereo/MB/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/stereo/MB/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/stereo/MB/crf0_lvl2_best.cpt" --multi-level-output --bp-inference sub-exp --with-bn --input-level-offset 1 --output-level-offset 1 --im0 "data/im0.png" --im1 "data/im1.png"
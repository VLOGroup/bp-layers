#/bin/sh
cd ops/lbp_stereo
python setup.py install
cd ../../

python main_stereo.py --model bp+ms+h --checkpoint-unary "data/params/stereo/BP+MS (H)/unary_best.cpt" --checkpoint-matching "data/params/stereo/Kitti/matching_lvl0_best.cpt" "data/params/stereo/Kitti/matching_lvl1_best.cpt" "data/params/stereo/Kitti/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/stereo/Kitti/affinity_best.cpt" --checkpoint-crf "data/params/stereo/Kitti/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/stereo/Kitti/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/stereo/Kitti/crf0_lvl2_best.cpt" --with-bn --bp-inference sub-exp --input-level-offset 0 --output-level-offset 0 --multi-level-output --im0 "data/000010_10_left.png" --im1 "data/000010_10_right.png"
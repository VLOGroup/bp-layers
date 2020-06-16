#/bin/sh
IM0="data/sf_0006_left.png"
IM1="data/sf_0006_right.png"

cd ops/lbp_stereo
python setup.py install
cd ../../

python main_stereo.py --model wta --checkpoint-unary "data/params/stereo/WTA (NLL)/unary_best.cpt" --checkpoint-matching "data/params/stereo/WTA (NLL)/matching_best.cpt" --with-bn --with-output-bn --im0 $IM0 --im1 $IM1

python main_stereo.py --model bp+ms --checkpoint-unary "data/params/stereo/BP+MS (NLL)/unary_best.cpt" --checkpoint-matching "data/params/stereo/BP+MS (NLL)/matching_lvl0_best.cpt" "data/params/stereo/BP+MS (NLL)/matching_lvl1_best.cpt" "data/params/stereo/BP+MS (NLL)/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/stereo/BP+MS (NLL)/affinity_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl2_best.cpt" --with-bn --bp-inference wta --multi-level-output  --im0 $IM0 --im1 $IM1

python main_stereo.py --model bp+ms+h --checkpoint-unary "data/params/stereo/BP+MS (H)/unary_best.cpt" --checkpoint-matching "data/params/stereo/BP+MS (H)/matching_lvl0_best.cpt" "data/params/stereo/BP+MS (H)/matching_lvl1_best.cpt" "data/params/stereo/BP+MS (H)/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/stereo/BP+MS (H)/affinity_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS (H)/crf0_lvl2_best.cpt" --multi-level-output --bp-inference sub-exp --with-bn  --im0 $IM0 --im1 $IM1

python main_stereo.py --model bp+ms+ref+h --checkpoint-unary "data/params/stereo/BP+MS+Ref (H)/unary_best.cpt" --checkpoint-matching "data/params/stereo/BP+MS+Ref (H)/matching_lvl0_best.cpt" "data/params/stereo/BP+MS+Ref (H)/matching_lvl1_best.cpt" "data/params/stereo/BP+MS+Ref (H)/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/stereo/BP+MS+Ref (H)/affinity_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS+Ref (H)/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS+Ref (H)/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/stereo/BP+MS+Ref (H)/crf0_lvl2_best.cpt" --checkpoint-refinement "data/params/stereo/BP+MS+Ref (H)/refinement_best.cpt" --with-bn --bp-inference sub-exp --output-level-offset 0 --multi-level-output --im0 $IM0 --im1 $IM1
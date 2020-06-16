#/bin/sh
IM0="data/frame_0019.png"
IM1="data/frame_0020.png"

cd ops/lbp_stereo
python setup.py install
cd ../../

python main_flow.py --model bp+ms+h --checkpoint-unary "data/params/flow/BP+MS (H)/unary_best.cpt" --checkpoint-matching "data/params/flow/BP+MS (H)/matching_lvl0_best.cpt" "data/params/flow/BP+MS (H)/matching_lvl1_best.cpt" "data/params/flow/BP+MS (H)/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/flow/BP+MS (H)/affinity_best.cpt" --checkpoint-crf "data/params/flow/BP+MS (H)/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/flow/BP+MS (H)/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/flow/BP+MS (H)/crf0_lvl2_best.cpt" --multi-level-output --bp-inference sub-exp --with-bn  --im0 $IM0 --im1 $IM1

python main_flow.py --model bp+ms+ref+h --checkpoint-unary "data/params/flow/BP+MS+Ref (H)/unary_best.cpt" --checkpoint-matching "data/params/flow/BP+MS+Ref (H)/matching_lvl0_best.cpt" "data/params/flow/BP+MS+Ref (H)/matching_lvl1_best.cpt" "data/params/flow/BP+MS+Ref (H)/matching_lvl2_best.cpt" --checkpoint-affinity "data/params/flow/BP+MS+Ref (H)/affinity_best.cpt" --checkpoint-crf "data/params/flow/BP+MS+Ref (H)/crf0_lvl0_best.cpt" --checkpoint-crf "data/params/flow/BP+MS+Ref (H)/crf0_lvl1_best.cpt" --checkpoint-crf "data/params/flow/BP+MS+Ref (H)/crf0_lvl2_best.cpt" --checkpoint-refinement "data/params/flow/BP+MS+Ref (H)/refinement_best.cpt" --with-bn --bp-inference sub-exp --output-level-offset 0 --multi-level-output --im0 $IM0 --im1 $IM1
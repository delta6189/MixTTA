#!/usr/bin/env bash
### ─── MixTTA: single entry point for all ImageNet-C experiments ───────────────
# Every experiment is launched through this script.
### ───────────────────────────────────────────────────────────────────────────

### Core configuration ###
GPU=${GPU:-0}
METHOD=${METHOD:-deyo}            # tent | eata | sar | deyo | recap_plpd | LinearTCA | no_adapt
MODEL=${MODEL:-vitbase_timm}            # vitbase_timm | resnet50_gn_timm
EXP=${EXP:-normal}                      # normal | mix_shifts | label_shifts | bs1
SEED=${SEED:-2024}
PLUGIN_MIXTTA=${PLUGIN_MIXTTA:-False}    # plugin_mixtta: False (baseline) | True (with MixTTA)

# LinearTCA can be used in two ways:
#   (1) standalone           -> METHOD=LinearTCA
#   (2) attached to a method -> ADD_TCA=True (reports "<METHOD>+LinearTCA", i.e. LinearTCA+)
# Per the paper, LinearTCA+ is LinearTCA on the best baseline: METHOD=recap_plpd ADD_TCA=True.
ADD_TCA=${ADD_TCA:-False}               # False | True

### Data paths ###
DATA=${DATA:-'/path/to/imagenet/val'}     # clean ImageNet val (EATA Fisher)
ROOT=${ROOT:-'/path/to/ImageNet-C'}       # ImageNet-C root

### MixTTA hyper-parameters ###
LAYER_TYPE=${LAYER_TYPE:-LoRAFC}        # LoRAFC (low-rank) | FC (full-rank)
INIT_TYPE=${INIT_TYPE:-xavier}          # xavier | orthogonal | kaiming
R=${R:-4}                               # low-rank subspace dimension
ETA=${ETA:-0.9}                         # Spectral Projection strength (0~1)
DECOUPLE_PROJ=${DECOUPLE_PROJ:-True}    # Decoupling Projection on/off

### Layer targeting (model-dependent defaults) ###
if [ "$MODEL" == "vitbase_timm" ]; then
    TENT_TARGET_BLOCKS=${TENT_TARGET_BLOCKS:-"0 1 2 3 4"}
    TENT_TARGET_LAYERS=${TENT_TARGET_LAYERS:-"0"}
    TENT_TARGET_NORMS=${TENT_TARGET_NORMS:-"1 2"}
    MIXTTA_TARGET_BLOCKS=${MIXTTA_TARGET_BLOCKS:-"0 1 2 3 4"}
    MIXTTA_TARGET_LAYERS=${MIXTTA_TARGET_LAYERS:-"0"}
    MIXTTA_TARGET_NORMS=${MIXTTA_TARGET_NORMS:-"2"}
else # ResNet
    TENT_TARGET_BLOCKS=${TENT_TARGET_BLOCKS:-"0 1 2 3"}
    TENT_TARGET_LAYERS=${TENT_TARGET_LAYERS:-"0 1 2 3 4 5"}
    TENT_TARGET_NORMS=${TENT_TARGET_NORMS:-"1 2 3 4"}
    MIXTTA_TARGET_BLOCKS=${MIXTTA_TARGET_BLOCKS:-"0 1 2"}
    MIXTTA_TARGET_LAYERS=${MIXTTA_TARGET_LAYERS:-"0 1 2 3 4 5"}
    MIXTTA_TARGET_NORMS=${MIXTTA_TARGET_NORMS:-"1 2 3 4"}
fi

echo "GPU=$GPU  Method=$METHOD  Model=$MODEL  Exp=$EXP  Seed=$SEED  PluginMixTTA=$PLUGIN_MIXTTA"

python -u main.py \
    --method $METHOD --model $MODEL --exp_type $EXP --seed $SEED --gpu $GPU \
    --data "$DATA" --data_corruption "$ROOT" \
    --Add_TCA $ADD_TCA \
    --plugin_mixtta $PLUGIN_MIXTTA \
    --layer_type $LAYER_TYPE --init_type $INIT_TYPE \
    --r $R --alpha $R --eta $ETA --decouple_proj $DECOUPLE_PROJ \
    --tent_target_blocks $TENT_TARGET_BLOCKS \
    --tent_target_layers $TENT_TARGET_LAYERS \
    --tent_target_norms $TENT_TARGET_NORMS \
    --mixtta_target_blocks $MIXTTA_TARGET_BLOCKS \
    --mixtta_target_layers $MIXTTA_TARGET_LAYERS \
    --mixtta_target_norms $MIXTTA_TARGET_NORMS

python tools/train.py --model simplenet

sed -i 's/bottle/carpet/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/carpet/grid/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/grid/leather/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/leather/tile/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/tile/wood/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/wood/cable/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/cable/capsule/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/capsule/hazelnut/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/hazelnut/metalnut/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/metalnut/pill/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/pill/screw/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/screw/toothbrush/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/toothbrush/transistor/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

sed -i 's/transistor/zipper/' ./src/anomalib/models/simplenet/config.yaml
python tools/train.py --model simplenet

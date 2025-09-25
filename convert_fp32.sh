
# Convert checkpoints from 000500 to 003000 in increments of 500 for multiple models
for model in 01-08-01 01-09-48 01-11-21 22-25-43 22-25-50; do
    echo "Processing model: $model"
    for step in 000500 001000 001500 002000 002500 003000; do
        DIR=/home/khemoo/simlingo/models/${model}/checkpoints/epoch=000-step=${step}.ckpt
        echo "Converting checkpoint: $DIR"
        python $DIR/zero_to_fp32.py $DIR $DIR
    done
    echo "Completed model: $model"
    echo "---"
done
Script does distillation on tasks of the GLUE dataset. Might need modification for non GLUE tasks. To run, I'm currently using

```
python distill_glue.py \
  --task_name rte \
  --teacher_model /scratch/pb2276/GLUE-pretrain/out_p2_l/rte \
  --student_model /scratch/pb2276/GLUE-pretrain/out_p2/rte/checkpoint-40 \
  --do_train \
  --do_eval \
  --gamma 1 \
  --max_seq_length 128 \
  --per_device_train_batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir distill_pure/rte/ \
  --save_strategy epoch \
  --eval_strategy epoch
```

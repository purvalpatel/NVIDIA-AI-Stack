```
!echo "@GSEKaFl9_nRN" | sudo -S docker run --rm \
  --runtime=nvidia \
  --gpus '"device=0,1"' \
  --shm-size=16GB \
  -v /home/nuvo_admin/.purval/ephemeral/models/llama3-8b-instruct-hf/original:/home/nuvo_admin/.purval/input_model -v TRTLLM_CKPT_DIR:/home/nuvo_admin_/.purval/output_model \
  -u $(id -u) \
  $NIM_IMAGE \
  python3 app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /input_model \
  --output_dir /output_dir/trtllm_ckpt \
  --dtype bfloat16
```

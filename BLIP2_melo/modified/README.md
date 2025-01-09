### 1. ValueError("You have to specify pixel_values")
```shell
# During the test for text locality, no image is provided
Traceback (most recent call last):
  File "run_VL.py", line 96, in run
    trainer.run_edit()
  File "/home/hy/yulang/BLIP2_melo/multimodal_trainer.py", line 67, in run_edit
    post_base_outputs = self.alg.get_output(eval_batch["loc"], lora_block_mapping)
  File "/home/hy/yulang/BLIP2_melo/algs/lora_blip.py", line 137, in get_output
    outputs = self.model.model(input_ids=input_ids, pixel_values=None, attention_mask=attention_mask)
  File "/home/hy/miniconda3/envs/melo_v2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hy/miniconda3/envs/melo_v2/lib/python3.8/site-packages/transformers/models/blip_2/modeling_blip_2.py", line 1690, in forward
    vision_outputs = self.vision_model(
  File "/home/hy/miniconda3/envs/melo_v2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hy/miniconda3/envs/melo_v2/lib/python3.8/site-packages/transformers/models/blip_2/modeling_blip_2.py", line 536, in forward
    raise ValueError("You have to specify pixel_values")
ValueError: You have to specify pixel_values
```


### Solution
Replace original blip2 model
```shell
site-packages/transformers/models/blip_2/modeling_blip_2.py
```
with
```shell
./BLIP2_melo/modified/modeling_blip_2.py
```

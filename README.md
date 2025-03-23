# replace layernorm with tanh
Add modifications for changing layernorm to dynamic tanh according to https://jiachenzhu.github.io/DyT/ and https://arxiv.org/abs/2503.10622

# added some more parameters to make it easier to change hyper-parameters under commandline
```
usage: gpt_tanh.py [-h] [--load LOAD] [--save SAVE] [--no-train] [--replace_ln_with_tanh REPLACE_LN_WITH_TANH] [--max_iters MAX_ITERS] [--eval_interval EVAL_INTERVAL] [--learning_rate LEARNING_RATE] [--eval_iters EVAL_ITERS]
                   [--n_embd N_EMBD] [--n_head N_HEAD] [--n_layer N_LAYER] [--dropout DROPOUT] [--block_size BLOCK_SIZE] [--batch_size BATCH_SIZE] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --load LOAD
  --save SAVE
  --no-train
  --replace_ln_with_tanh REPLACE_LN_WITH_TANH
  --max_iters MAX_ITERS
  --eval_interval EVAL_INTERVAL
  --learning_rate LEARNING_RATE
  --eval_iters EVAL_ITERS
  --n_embd N_EMBD
  --n_head N_HEAD
  --n_layer N_LAYER
  --dropout DROPOUT
  --block_size BLOCK_SIZE
  --batch_size BATCH_SIZE
  --device DEVICE
```

# Original README.md below

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT

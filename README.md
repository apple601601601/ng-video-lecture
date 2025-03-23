This repo is a fork of Andrej Karpathy's ng-video-lecture repo and I made some changes for playing around and experiments.

# Updates:
### Replace layernorm with tanh
Add modifications for changing LayerNorm to Dynamic tanh according to https://jiachenzhu.github.io/DyT/ and https://arxiv.org/abs/2503.10622
(Didn't see much difference in terms of results and efficiency, probably the model is too small to show significant differences.)

### Added some more parameters to make it easier to change hyper-parameters under commandline
```
python ./gpt_ln2tanh.py -h
usage: gpt_ln2tanh.py [-h] [--load LOAD] [--save SAVE] [--no-train] [--replace_ln_with_tanh REPLACE_LN_WITH_TANH]
                      [--max_iters MAX_ITERS] [--eval_interval EVAL_INTERVAL] [--learning_rate LEARNING_RATE]
                      [--eval_iters EVAL_ITERS] [--n_embd N_EMBD] [--n_head N_HEAD] [--n_layer N_LAYER] [--dropout DROPOUT]
                      [--block_size BLOCK_SIZE] [--batch_size BATCH_SIZE] [--device DEVICE] [--randseed RANDSEED]

options:
  -h, --help            show this help message and exit
  --load LOAD           Load a model from a file
  --save SAVE           Save a model to a file, will be in "weight" folder
  --no-train            Do not train the model, inference only
  --replace_ln_with_tanh REPLACE_LN_WITH_TANH
                        Replace LayerNorm with Dynamic tanh, default is False
  --max_iters MAX_ITERS
                        Number of iterations to train, default is 5000
  --eval_interval EVAL_INTERVAL
                        How many number of iterations between evaluation, default is 100
  --learning_rate LEARNING_RATE
                        Learning rate for training, default is 3e-4
  --eval_iters EVAL_ITERS
                        Number of iterations to average loss during evaluation, default is 200
  --n_embd N_EMBD       Embedding dimension, default is 384
  --n_head N_HEAD       Number of heads, default is 6
  --n_layer N_LAYER     Number of layers, default is 6
  --dropout DROPOUT     Dropout rate, default is 0.2
  --block_size BLOCK_SIZE
                        Block size, default is 128
  --batch_size BATCH_SIZE
                        Batch size, default is 64
  --device DEVICE       Device to use, default is cuda if available
  --randseed RANDSEED   random seed to use, default to 1337
```


# Example:

## train with default hyper parameters and LayerNorm 
```
python ./gpt_ln2tanh.py
===Use LayerNorm===
10.739777 M parameters
step 0: train loss 4.3205, val loss 4.3128
step 100: train loss 2.5131, val loss 2.5166
step 200: train loss 2.3533, val loss 2.3809
step 300: train loss 2.0839, val loss 2.1292
step 400: train loss 1.8718, val loss 1.9936
step 499: train loss 1.7531, val loss 1.9010
Saving model to model_nanogpt_ln_2025-03-22 22_32_45.668254.bin
====Seconds used for training: 164.88862347602844====

Sirt, Myraiusirves
Ated the his this hilds promp.
Espe eurt, year thre: nort hanst Yet forthince theed goodioth,
As amtand anoualt would, araig its mus.
Whe tOmand hre twell be that seedied, that bentle the it you
nnown urneds, hapine hied fin,
I well burd heaflow, cond the you powith,
Minionds then foundedest rouns of fly hay.

3 KING HENRY:

E, they; foigh set peak? my as but Lirial!s, dod
Clear. Mared, not. Slreat thheir-ble practers!

&ZABETHUM:
There ber bayy sandy.

HiWhat tead:
Tish nothy
====Seconds used for inference: 19.585687160491943====
```

## train with default hyper parameters and replace LayerNorm with Dynamic tanh
```
python ./gpt_ln2tanh.py --replace_ln_with_tanh True
===Replace LayerNorm with tanh===
10.739789 M parameters
step 0: train loss 4.2427, val loss 4.2421
step 100: train loss 2.4777, val loss 2.4953
step 200: train loss 2.4149, val loss 2.4437
step 300: train loss 2.3558, val loss 2.3771
step 400: train loss 2.2923, val loss 2.3306
step 499: train loss 2.2492, val loss 2.2914
Saving model to model_nanogpt_tanh_2025-03-22 22_55_23.033295.bin
====Seconds used for training: 167.00086760520935====

Sirth nore usircer
Et! heat, ird, the dive Rath sale e eally y.


NOFL:
EVrd no stay tofo, thice thede g dait balds am tobld oualt wiI Ensty is icow,
Mathat tOnale hrgh w thebre hatuse ourded; loubend,
I To ithyoud my Caucked whow ine heed f poqure penouignhe,
And, co Pan'dwhend thithe, teis rst hen founde e torout nthully hay. pe spowoulbuninnt,

Pay; maige se spthathe, aporour dor has, dod
Cle Witra ed, umy on To thitheing he pemaress!
O&G it hat herded ber baveand theue, Whak te, wisind n ano
====Seconds used for inference: 17.491108655929565====
```

## Other command samples
```
# train with a different embedding dimension
python ./gpt_ln2tanh.py --n_embd 384
python ./gpt_ln2tanh.py --replace_ln_with_tanh True --n_embd 384

# train with a different embedding dimension and block size
python ./gpt_ln2tanh.py --n_embd 384 --block_size 256
python ./gpt_ln2tanh.py --replace_ln_with_tanh True --n_embd 384 --block_size 256

# inference with a model that is trained with different embedding dimension and block size
python ./gpt_ln2tanh.py --n_embd 384 --block_size 256 --no-train --load "model_nanogpt_ln_2025-03-22 19_28_20.428231.bin"
python ./gpt_ln2tanh.py --replace_ln_with_tanh True --n_embd 384 --block_size 256 --no-train --load "model_nanogpt_tanh_2025-03-22 19_34_35.128400.bin"

# inference with a model that is trained with default embedding dimension and block size
python ./gpt_tanh.py --no-train --load "model_nanogpt_ln_2025-03-22 22_32_45.668254.bin"
python ./gpt_ln2tanh.py --no-train --load model_nanogpt_ln2tanh.bin --replace_ln_with_tanh True  
```

---

# Original README.md below
# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT

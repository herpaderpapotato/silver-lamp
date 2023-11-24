about the model(s)


required:
* models must normalize images themselves from 0-255 to whatever they need
* models must take input of frame count, x, y, channels. At the moment frame count is fixed per model (60)
* don't shrink a post LSTM layer to less than the frame count

desirable:
* models should resize images themselves on the GPU
    * probably, I don't know what is more of a bottleneck, moving more, bigger tensors to GPU and potentially consuming VRAM. or CPU capacity.


concept:

Just like any "fine tune this pretrained model guide":
1. We get a pretrained model.
2. We slap a dense layer on it

but then

3. We wrap it in a time distributed
4. We use bidirectional LSTM
5. We arbitrarily chain some dense layers and dropouts, down to a final dense layer of the frame count side

the results:

A brute force approach to linear movement estimation in sequential images.



scratchpad

previously trained frame skipping (simulate 30fps data vs 60fps)

may need to train on non linear frames to make the model less linear or more adaptable. May even mix up videos in a sequence or find matching trajectories across videos and mash together

The training data is all created by hand and does not use peak to peak linear values (sawtooth). It is not using community or commercially generated funscripts. The goal is to generate base funscripts to enhance throughput of funscript creators, to help them compete with AI, not to replace them with AI.

Currently only 'post trained' the efficientnetv2s model. The XL model should score higher. Also finetuning the base model may yield positive results.




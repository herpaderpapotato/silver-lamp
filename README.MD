ML for generating funscripts. Slow.

Tested with wsl2, python3.10 and nvidia gpu. Also tested with cpu (remove [and-cuda] from requirements.txt) but it's slower.

Standard python + github process to use this
* probably make a conda environment if you have/use it.
* git clone https://github.com/herpaderpapotato/silver-lamp
* pip install -r requirements.txt
* python inference.py --video "video path here"

By default I've got it set to download the model from https://huggingface.co/herpaderpapotato/sixty_small_body_ryhthm_time_bidirectional

inference.py will load the video in opencv.
* b to skip forward 10 minutes
* v to skip forward 1 minute
* c to skip back 1 minute
* x to skip back 10 minutes
* z to skip back 10 seconds
* h to hold on the current frame
* p to start making predictions
* mouse click to select a point for "point of interest" predictions
* [ to shrink the poi window
* ] to grow the poi window
* , to shrink the frame delay (faster rendering)
* . to make the frame delay longer
* q to quit and export the collected predictions to ./predictions folder

This isn't doing batching, it's not as quick as it could be.
* It could be even quicker if it just did one type of prediction. 
* Smaller video files is also quicker (1080p to start)
* cuda opencv would probably speed things up more again

It's not as accurate as it could be
* probably poi seems most accurate approach of prediction (tight crop)
* more training can be done on the model
* more dataset would help, including some of the less obvious movements
* the base model hasn't been finetuned
* smoothing/overlap can also improve the output
* the range of the outputs can/should be tuned afterwards as it can be a bit low/accurate.

At the moment it is only suitable (barely) for taking some of the tedium out of the process, especially when it comes to obvious and repetitive movement.

Maybe probably migrates to pytorch so that GPU inference can be done in windows. Or tflite for faster cpu in windows. Or onnx. 

And to test/do more in windows for those not already comfortable with wsl.

[Stuff I had to do on a fresh WSL install to get it going](TROUBLESHOOTING.MD)


Currently looking for ways that don't require as much handholding. i.e. this works, but isn't broad enough. A combination of my training data, model architecture, or approach are inadequate to a "label any video" goal.

This could make great funscripts if you:

1. had a lot more training data
2. train separate models for different positions
3. use a workflow (position identification model?) that uses the different models based on position
4. had an ongoing desire to continually train and tune that over time as inaccuracies were found

But I'd also like to see a model handle longer predictions of 2 to 20 seconds of video per prediction to enable smoother movement. As well as having the model take an initial position as an input.

Alternate model inputs being looked at are things like:
* canny outputs
* edge enhanced outputs
* using images segmented by skin detection
* bit packing image masks to lower parameter count
    * e.g. a 256 x 256 x 1 image that is a 1 bit mask, *could* pack into a float 32, reducing the parameters by a factor of 32, BUT it's potentially harder for the model to learn the data given the bitwise nature of it.
* using images segmented by segmentation models

They all take time, energy and effort. The most interesting path I have at the moment is cloning the [DeepLabv3](https://github.com/tensorflow/models/blob/master/research/deeplab/README.md) architecture, and instead of the final segmentation output as an input to the model, using an earlier layer as a feature extraction layer.
I would like to add another experiment in this project (folder experiments/object_detection) for detecting UAVs captured with event cameras and RGB by using the image formats called "xt_my", "yt_mx" and "cstr3" in this repository.
"xt_my" uses histogram of events for both polarities in two channels and the normalized mean y coordinate in the third channel. "yt_mx" works the same way but uses mean x instead.

I would like to first try to use a two custom CNNs that uses xt_my and yt_mx as inputs and learns a heatmap in that as a head from the concatenated features. Then I would like to add another head that learns the boxes coming from the YOLO dataset (available in datasets/FRED).

There are already several experiments that use the FRED dataset in this project, but they are mainly focused on forecasting and other things. But there should be a lot of helper code available.

Make it easy to extend and use other backbones and image formats later so one can add more image representations later and test with ResNet, ViTs etc later.

Make sure that you understand the image formats we use so that it is possible to learn the heatmap correctly directly in the xt- and yt-space. One axis is always the time, but the spatial dimension is always "correct" in the image so the time is in the y-axis for xt_my and the x-axis for yt_mx.
I guess that we need to translate the xy boxes into xt and yt so that they cover the entire t axis always and mark that as the correct heatmap for that training sample. Not sure how this works though, so please fill in any of my knowledge gaps here.

I need to manually run scripts/render_fred_splits.py then to generate 33.333 ms long images that matches the framerate of the samples, but please check that this is correct also. There are some examples of how this scripts are used for various window lengths, but in this case we should make extra sure that the window matches in time so that it matches the current box exactly.

Also provide documentation in a file like the other experiments do.



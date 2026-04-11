I would like to add another experiment in this project that performs drone forecasting using curve fitting in images contructed from event camera data.

The forecasting task is defined in the docs/fred_paper and a lot of the metrics, dataset etc are implemented in previous experiments in the experiments folder.

This new experiment/curve_fit_forecasting should be based on the script/fit_event_xy_trace.py that uses image formats xt_my and yt_mx to fit a curve in the xy space, making it possible to predict the future movement of the drones.

This future curve should be used to predict the future boxes and center of such boxes by the curve and past boxes (multiple such dataset classes are available in the project now but we should probably create a new one for this specific setting).

The curve fitting should be extended to be guided by historic boxes so that we do not fit a curve to some insect or shadow of the drone. They can for example be used to filter out regions in the xt and yt space where we consider to use points to fit (with some slack because the boxes are not always spot on). They can also be used to not trust a curve that is too far off the curve created from history boxes if that makes sense.

We should probably add some training later to be able to predict the future box sizes and not just the centers based on historic boxes and images (and drone label if allowed in the forecasting task?), but we can just use some kind of simple mean box size taken from the past boxes for now.

The setup requires some parameters to specify how many milliseconds the input images were created from. It should be 400 ms as default. The forecasting task can be either 400 ms or 800 ms but any forecasting horizon should be possible to specify in the config.

There is code available to match images with historic boxes as used by the dataset classes, so this can be reused.

Make it possible to provide parameters to the curve fitting process in the config as well. We will probably have to tweak those parameters and possibly learn them based on the training set as well given the performance of matching curves that are OK based on the historic boxes. We might be able to find some optimal or really good settings for thresholds etc in this way.

Feel free to add or modify things if anything I've mentioned seems off or doesn't make sense, but please explain why in such cases.


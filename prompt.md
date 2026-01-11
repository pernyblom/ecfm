I would like to create a foundation model (a transformer) over event camera data which is a sequence of (x, y, t, p) tuples.

The foundation model should operate over aggregated event counts in a spatio-temporal region (x, y, t, dx, dy, dt) and the summation can be performed over the xy, xt or yt.

The histogram is then max-normalized, but the total number of events are kept so it is possible to re-create the real histogram. It is then made into a patch by resizing the normalized histogram into a rectangular "patch" that is then used to create the embedding from.

Somehow, for example via spatio-temporal embedding or via the patch data, the position and size of the region should be included in the embedding either by absolute values or relative values when using multiple tokens/embeddings.

The training of the foundation model could be done with a MAE approach where one can use a novel(?) masking approach that can mask any set of custom spatio-temporal regions and try to reconstruct it given another non-overlapping set of regions.

Not sure if this approach even will work, but the idea is to get a very flexible foundation model out that can support multiple levels of time scales so that it is possible to use both for long integration times and very short ones as well.

Do you think that this project is possible? I would like to try with very small models first to be able to iterate fast.

The augmentation I would like to use has to be performed in the event data so we need some event rotation routines for this, but I think that if the patches dx, dy, and dt are varied a lot, I don't think that we need any temporal augmentations except maybe some time warp that non-linearly changes the dynamics without changing the total length of a sequence.

I plan to use some event data from a wide range of datasets with both stationary and moving sensors.

I also want to use pytorch for all this.

Can you create a document of a possible design for all this and create a pytorch project in the current folder with a suitable git repo structure?
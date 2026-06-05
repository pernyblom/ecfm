
# Introduction draft

Event cameras can see fast things. 
Propellers for example, and it should be possible to predict multirotor UAVs better based on events compared to classical prediction models.

However, Kalman/physical filter performance strong on forecasting outperforms raw ML methods on FRED dataset.

We investigate a hybrid approach where we combine a strong constant kalman filter performance with a ML for the residuals/acceleration.

Extend previous results with boxes and get SOTA results on FRED when using various image representations of events and combinations thereof.

Care is taken to extend the learned model to OOD performance by not directly providing any absolute positions by using cutouts around the boxes and relative position inputs for the box coordinates.

However, there is a strong correlation with velocity to acceleration so we also implemented a decorrelation method.

Also bias towards acceleration down more due to gravity and that most scenes have a horizontally directed camera. Trying to remove this depends on the OOD target.



# What to include:
- Intro
- Related work
- Short description of FRED dataset
- Description of baseline kalman and how it was optimized
- Description of how correlation is analyzed and linear regression setup
- Description of experiment setup with image formats used and motivation for cutouts etc. and try to avoid any position info in the image formats (such as mean y in xt cutout)
- Results of correlation from pos/vel to acceleration with training results linear regression (without history boxes, only kalman filter state), images showing arrows for pos/vel for train/test split
- Results with combinations of image inputs on raw and decorrelated split.
- Comparison with SOTA
- Possible ablations
    + Size of cutouts
    + Temporal bin counts and/or size of cutout in time dimension (all images currently rendered with bin count 224)
    + Smaller CNN (might generalize better?)
    + Inclusion of covariance features from kalman filter
- Discussion


# Main contributions
- Correlation analys, decorrelation method and discussion with OOD in mind
- SOTA results on FRED with boxes


# Preliminary findings
- Decorrelation method seem to work but is not perfect. Still possible to get better residuals from vel/pos with linear regression, although with a worse miou.
- The simple extension of kalman for sizes compared to the paper from Finland immediately gives SOTA on box results compared to the original FRED paper (miou, box ADE/FDE etc.). We can't beat them on center ADE/FDE with our kalman filter.
- But it is pretty easy to get SOTA results the non-decorrelated dataset when including event images even when using cutouts, relative box positions, no image features giving away position etc.
- The pre-rendered event frames that comes with the dataset gives the best result on the tested image formats on the raw dataset (event_frames). We tested with cstr2, xt, yt.
- On the decorrelated subset we get better results with cstr2. No idea why it is better but the cstr2 uses normalized mean timestamps which makes it possible to see the gradient of movement instead of the binary-like event_frames. Why or if this matters at all we don't know.
- We haven't yet tested much combinations of image formats so we don't know yet what works best. We got some very early improved results with event_frames/cstr2 + xt/yt but we need to test this more and check if RGB is useful to include as well.


# Choices and limitations
- Resized input frame to 640x360 for all image sizes
- Always using the train_eval_split.enabled: true and a fraction of 0.2
- Cutouts only (motivated by getting rid of positional info and computational benefits)
- Always using 64 as cutout size (which would be 128 in the original frame size right?)
- Relative positions only in box coords (history_feature_mode: relative)
- Position information removed from image formats (using xt, yt instead of xt_my etc.)
- Temporal dim in xt/yt variants is always size 64
- All tracks 1.2 s, even the 0.4 ms prediction
- Decorrelation set mean_accel_weight: 1.0e-4 to mildly remove acceleration bias
- Resnet-18 used only. It would be interesting to see if a much smaller CNN works as well in ablation
- Always predicting size residuals
- filter_state_center_position_normalization: frame_centered
- Not using any covariance features from kalman filter
- Always using optimized kalman filter as the state features





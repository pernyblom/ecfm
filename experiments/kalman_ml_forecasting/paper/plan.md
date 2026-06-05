
Event cameras can see fast things. 
Propellers for example, and it should be possible to predict multirotor UAVs better based on events compared to classical prediction models.

However, Kalman/physical filter performance strong on forecasting outperforms raw ML methods on FRED dataset.

We investigate a hybrid approach where we combine a strong constant kalman filter performance with a ML for the residuals/acceleration.

Extend previous results with boxes and get SOTA results on FRED when using various image representations of events and combinations thereof.

Care is taken to extend the learned model to OOD performance by not directly providing any absolute positions by using cutouts around the boxes and relative position inputs for the box coordinates.

However, there is a strong correlation with velocity to acceleration so we also implemented a decorrelation method.



What to include:
- Short description of FRED dataset
- Description of baseline kalman and how it was optimized
- Analysis of correlation from pos/vel to acceleration
- 




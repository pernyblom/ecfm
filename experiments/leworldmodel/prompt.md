I would like to implement a variant of the LeWorldModel (paper source is in experiments/leworldmodel/paper) but without actions applied to the The Florence RGB-Event Drone Dataset (FRED).
The current project has explored this dataset before and has several experiments where one uses various event data image representations to perform forecasting or extracting pose dynamics.
This new experiment (in folder experiments/leworldmodel) that I want to you implement will try to perform LeWorldModel-inspired self-supervised learning on those image representations for prediction and the forecasting task (described in the FRED paper available in folder docs/fred_paper) as a possible downstream task.

The training is supposed to work like this:
- Use a set of image representations (xt_my, yt_mx, cstr3 etc) and put those into a CNN to get a latent space
- Use the latent space to predict the future latent space (computed from the future image representations)
- What type of REG to use (simple (what do you suggest?), VicREG, SIGRREG (used in the paper))
- Use the latent loss + LeWorldModel anti-collapse loss to create a nice representation for prediction

I would like the implementation to support various variations that can be set with a config (like all the other experiments do in this project through a yaml file):
- Choice of encoder (custom small CNN, resnet-18, ViT-tiny etc.)
- Predictor type (simple MLP, ViT)
- How many previous frames to use in the encoder
- Other important parameters that you consider to be good to have in the config

The representations are created with a separate script (scripts/render_fred_splits.py).

I would like to start with a single set of frames (one timestep) as input with xt_my, yt_mx and cstr3 in the encoder part with a small CNN. The predictor can be a simple MLP for starters just to see if it even works. 

We have to make sure that each frame matches a next frame so that there is no large gap. Only use frames with UAVs in them.


Also implement a simple version of forecasting using the latent variables + input boxes predicting future boxes. Make it possible to inline such downstream tasks to check what scores we get on forecasting. There are some stuff implemented already in the experiments/forecasting so we can possibly reuse some of that code.

The leworldmodel/prompt.md contains just this initial prompt so you can ignore that.


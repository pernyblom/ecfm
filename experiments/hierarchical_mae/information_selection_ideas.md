Yes. In your setting, I would avoid a generic image-noise metric such as PSNR or variance and instead ask:

> **Given the same number of events, how much spatio-temporal structure does this voxel contain?**

That distinction is important because event count is already working for you, and many naive metrics will mostly rediscover event count.

A particularly useful metric would be a **structure-vs-shuffled-noise score**. Suppose your voxel is a histogram

\[
V(x,y,t)
\]

with \(N=\sum V(x,y,t)\) events. Construct a noise/null version by keeping \(N\) fixed but randomly permuting the event locations/times within the voxel. Then measure something sensitive to local structure on both the real voxel and the shuffled one. For example, local autocorrelation:

\[
A(V)=
\sum_{\Delta\in\mathcal N}
\frac{\sum_i (V_i-\bar V)(V_{i+\Delta}-\bar V)}
{\sum_i(V_i-\bar V)^2+\epsilon}
\]

where \(\mathcal N\) could include spatial and temporal neighbors,

\[
(1,0,0),\;(0,1,0),\;(0,0,1)
\]

and perhaps diagonals.

Then define

\[
S_{\text{structure}}
=
A(V)-E[A(V_{\text{shuffle}})].
\]

A voxel containing an edge moving through space-time will have strong correlations between neighboring bins. Uniform/random background activity tends not to. Crucially, because you shuffle **the same events**, the score is much less dependent on event count.

An even better version is to standardize it:

\[
Z =
\frac{A(V)-\mu_{\text{shuffle}}}
{\sigma_{\text{shuffle}}+\epsilon}.
\]

Then you're essentially asking:

> "How unlikely is this voxel under a random-event hypothesis?"

That seems very aligned with your MAE token-selection problem.

There are several other metrics I'd experiment with alongside it:

- **Entropy / entropy deficit.** Normalize the histogram \(p_i=V_i/N\) and calculate
  \[
  H=-\sum_i p_i\log p_i.
  \]
  Randomly spread events generally give high entropy; concentrated structure gives lower entropy. A normalized score such as
  \[
  1-\frac{H}{\log K}
  \]
  where \(K\) is the number of bins, measures concentration. The caveat is that a hot/noisy pixel can also have low entropy.

- **Spatio-temporal total variation.**
  \[
  TV(V)=\sum |\nabla_x V|+|\nabla_y V|+\lambda|\nabla_tV|.
  \]
  Useful structure often produces coherent surfaces in \(x,y,t\), although raw TV can also become large with salt-and-pepper noise. I'd normalize it by event count.

- **Spectral flatness.** Take a 3-D FFT of the voxel. White-ish noise has a comparatively flat spectrum, while coherent motion/edges produce concentrated spectral energy:
  \[
  SF =
  \frac{\exp(\frac1K\sum_k \log(P_k+\epsilon))}
       {\frac1K\sum_k P_k+\epsilon}.
  \]
  \(SF\approx1\) means noise-like; lower \(SF\) means more structured. This is attractive because it gives a literal "noise-likeness" score.

- **Local neighbor support.** For every event/bin, measure how many neighboring spatio-temporal bins also contain events. Something like
  \[
  C =
  \frac{1}{N}\sum_i V_i
  \mathbf 1\left[\sum_{j\in\mathcal N(i)}V_j>0\right].
  \]
  Isolated events score poorly; events forming trajectories/surfaces score highly. This one is extremely cheap.

For event-camera data specifically, I think **neighbor support / spatio-temporal coherence** is likely to give you the best compute-to-benefit ratio. Real edges don't just generate many events; they tend to generate locally coherent collections of events:

\[
(x,y,t)
\rightarrow
(x+\Delta x,y+\Delta y,t+\Delta t).
\]

Background activity/noisy pixels are much less likely to produce that geometry.

You could therefore replace your current

\[
\text{score}(V)=N
\]

with something as simple as

\[
\boxed{
\text{score}(V)
=
\log(1+N)\;
C(V)
}
\]

where \(C(V)\in[0,1]\) is the fraction of events having sufficient nearby spatio-temporal support.

Or, if you don't want low-event but very coherent voxels to be overly penalized,

\[
\boxed{
\text{score}(V)
=
\alpha\,\operatorname{norm}(\log(1+N))
+
(1-\alpha)\,\operatorname{norm}(C(V)).
}
\]

### One metric I'd test first

I'd actually make the coherence measure slightly more event-camera-specific. For each occupied bin at \((x,y,t)\), examine a small space-time neighborhood,

\[
[x-r,x+r]\times[y-r,y+r]\times[t-\tau,t+\tau].
\]

Calculate the number of other events/bins inside it:

\[
n_i = \sum_{j\in \mathcal N(i)}V_j.
\]

Then

\[
C(V)
=
\frac{
\sum_i V_i\,\min(n_i,c)
}{
cN
}.
\]

Here \(c\) is a saturation threshold, perhaps 3–10 depending on your binning.

This distinguishes nicely between:

```text
noise                       useful edge/motion

 .     .                       ###
      .                     ###
  .       .               ###
       .                ###
```

Both can have exactly the same number of events, but the right-hand voxel has much larger local support.

There's an additional trick that I think could be especially effective for your MAE setup: **rank by information beyond event count rather than replacing event count entirely**.

For example, estimate

\[
E[C\mid N]
\]

from your training data and use

\[
S(V)=C(V)-E[C\mid N].
\]

So a voxel gets selected if it is **more structured than a typical voxel containing the same number of events**. That prevents the selector from simply learning "more events = better."

If I were benchmarking this, I'd compare **event count**, **normalized entropy**, **3-D spectral flatness**, **local spatio-temporal support**, and **shuffle-normalized autocorrelation**. My guess is that local support will be the best cheap metric, while shuffle-normalized autocorrelation will be the cleanest principled definition of "less noise-like."

A particularly interesting next step would be to evaluate these not by whether they *look* structured, but by **MAE reconstruction loss / downstream performance as a function of selected-token percentile**. That would tell you whether your metric is actually identifying useful information rather than merely visually structured event patterns.
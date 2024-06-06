# Learnable Skip

Just some quick experimentation. I had an idea while reading [highway networks](https://arxiv.org/abs/1505.00387) that it might be nice to allow models with skip connections the comptelety "turn off" the skip connection. With current architectures, if a layer wants to override the residual stream. It has to learn to produce both the new features, and the inverse of the existing residual stream. By adding a learned (learned during training, static at inference time) gate parameter, it enables layers to learn to completely override the residual stream.

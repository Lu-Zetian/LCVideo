def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_original(pred_noise, timesteps, sample, alphas, sigmas):
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    pred_x_0 = (sample - sigmas * pred_noise) / alphas

    return pred_x_0
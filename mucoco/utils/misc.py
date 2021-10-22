def get_epsilon(step, max_e, min_e, warmup_steps, cooldown_steps, decay_function):
    if decay_function == "none" or max_e == min_e:
        return max_e
    elif decay_function == "linear":
        if step <= warmup_steps:
            return max_e
        elif step > warmup_steps and step <= cooldown_steps:
            return max_e - (max_e - min_e)*(step-warmup_steps)/(cooldown_steps-warmup_steps)
        else:
            return min_e
    elif decay_function == "rsqrt":
        if step <= warmup_steps:
            return max_e
        elif step > warmup_steps and step <= cooldown_steps:
            return max_e - (max_e - min_e) * (np.sqrt(cooldown_steps-warmup_steps+1)/(np.sqrt(cooldown_steps-warmup_steps+1) - 1)) * (1 - 1/np.sqrt(step - warmup_steps + 1))
        else:
            return min_e
    elif decay_function.startswith("poly"):
        p = float(decay_function[5:])
        if step <= warmup_steps:
            return max_e
        elif step > warmup_steps and step <= cooldown_steps:
            return max_e - (max_e - min_e) * ((cooldown_steps-warmup_steps+1) ** p/((cooldown_steps-warmup_steps+1)**p - 1)) * (1 - 1/(steps - warmup_steps + 1)**p)
        else:
            return min_e
    elif decay_function == "exponential":
        if step <= warmup_steps:
            return max_e
        elif step > warmup_steps and step <= cooldown_steps:
            expo = -(step - warmup_steps)/(cooldown_steps - warmup_steps) * np.log(max_e/min_e)
            return max_e * np.exp(expo) 
        else:
            return min_e
    elif decay_function == "step":
        if step <= warmup_steps:
            return max_e
        else:
            return min_e

# def plot_gd(losslist, filename="plot.png"):
#     import matplotlib.pyplot as plt

#     plt.clf()
#     print(losslist)
#     input
#     # for i, (loss1, loss2, gx, gy, sx, betas) in enumerate(plotdata):
#     plt.plot(losslist[0][-10:], losslist[1][-10:])
#     # plt.axhline(y=sx)
#     # plt.plot(gx, gy, 'ro')
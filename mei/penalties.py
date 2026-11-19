# def zero_mean():
#     def penalty(x):
#         mean_penalty = x.mean().pow(2)
#         range_penalty = po.regularize.penalize_range(x, (-1, 1))
#         return mean_penalty + range_penalty
#     return penalty

def just_range():
    return po.regularize.penalize_range

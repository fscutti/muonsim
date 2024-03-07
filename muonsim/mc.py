import numpy as np
from tqdm import tqdm


def uniform(num_samples, parameters):
    """Uniform distribution."""
    samples = []
    for idx in range(num_samples):
        s = []
        for _, p_range in parameters.items():
            s.append(np.random.uniform(*p_range))
        samples.append(s)
    return np.array(samples)


def metropolis_hastings(num_samples, likelihood, initial_sample, proposal_std, burning):
    """Metropolis-Hastings MCMC algorithm for 2D mean."""

    samples = []
    current_sample = initial_sample

    for idx in tqdm(range(num_samples)):
        proposed_sample = current_sample + np.random.normal(
            0, proposal_std, size=len(proposal_std)
        )

        acceptance_ratio = likelihood(proposed_sample)
        acceptance_ratio /= likelihood(current_sample)

        if np.random.rand() < acceptance_ratio:
            current_sample = proposed_sample

        if idx < burning:
            continue

        samples.append(current_sample)

    return np.array(samples)

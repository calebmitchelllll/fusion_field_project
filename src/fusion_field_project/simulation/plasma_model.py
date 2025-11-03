from scipy.constants import mu_0

def beta_estimate(pressure_Pa: float, B_T: float) -> float:
    """Toy plasma beta ~ (2*mu0*p)/B^2."""
    return (2*mu_0*pressure_Pa)/((B_T**2) + 1e-30)
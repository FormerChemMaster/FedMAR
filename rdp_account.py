import math

def compute_rdp(alpha, sigma, sensitivity):
    return alpha * sensitivity**2 / (2 * sigma**2)

def get_sigma(epsilon, alpha, sensitivity):
    return math.sqrt(alpha * sensitivity**2 / (2 * epsilon))
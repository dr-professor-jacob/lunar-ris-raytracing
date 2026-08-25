"""
Lunar Regolith Radio Material Model for Sionna-RT Ray Tracing
"""

import numpy as np

class LunarRegolithMaterial:
    """
    Models the electromagnetic dielectric and reflection characteristics
    of lunar regolith based on Apollo soil samples and ITU-R P.2040 dielectric models.
    """
    def __init__(self, relative_permittivity=2.7, loss_tangent=0.008, conductivity=1e-4):
        self.epsilon_r_real = float(relative_permittivity)
        self.loss_tangent = float(loss_tangent)
        self.conductivity = float(conductivity)
        
    def complex_permittivity(self, frequency_hz):
        """
        Calculates complex relative permittivity: epsilon = epsilon' - j * epsilon''
        where epsilon'' = epsilon' * loss_tangent + (sigma / (2 * pi * f * epsilon_0))
        """
        epsilon_0 = 8.8541878128e-12
        omega = 2.0 * np.pi * frequency_hz
        epsilon_imag = self.epsilon_r_real * self.loss_tangent + (self.conductivity / (omega * epsilon_0))
        return complex(self.epsilon_r_real, -epsilon_imag)

    def fresnel_reflection(self, incidence_angle_rad, frequency_hz, polarization='parallel'):
        """
        Computes Fresnel reflection coefficient for lunar regolith surface.
        :param incidence_angle_rad: Angle of incidence relative to surface normal (radians)
        :param polarization: 'parallel' (TM) or 'perpendicular' (TE)
        :return: Complex reflection coefficient R
        """
        eps_c = self.complex_permittivity(frequency_hz)
        theta_i = incidence_angle_rad
        
        # Snell's law: n1 sin(theta_i) = n2 sin(theta_t), n1 = 1 (vacuum)
        sin_theta_t = np.sin(theta_i) / np.sqrt(eps_c)
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
        cos_theta_i = np.cos(theta_i)

        if polarization == 'perpendicular':
            # TE Polarization
            R = (cos_theta_i - np.sqrt(eps_c) * cos_theta_t) / (cos_theta_i + np.sqrt(eps_c) * cos_theta_t)
        else:
            # TM Polarization (Parallel)
            R = (np.sqrt(eps_c) * cos_theta_i - cos_theta_t) / (np.sqrt(eps_c) * cos_theta_i + cos_theta_t)
            
        return R

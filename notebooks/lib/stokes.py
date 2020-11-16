import numpy
from scipy import special 
import bqplot.pyplot as plt
from ipywidgets import (interactive, Layout, HBox, VBox, Box, Label,
                        IntSlider, FloatSlider)


def voigt(gamma, x):
    """
    Calculates the Voigt function.
    """
    z = (x + 1j * gamma)
    return special.wofz(z).real / numpy.sqrt(numpy.pi)


def faraday(gamma, x):
    """
    Calculates the Faraday dispersion function.
    """
    z = (x + 1j * gamma)
    return special.wofz(z).imag / numpy.sqrt(numpy.pi)


def unno_rachkovsky(u, s0=1, s1=5, eta=20, a=0.05, g_eff=1, 
                    delta_ratio=1.5, gamma=numpy.pi/3, chi=0):
    """
    Calculates Stokes vector using Unno-Rachkovsky solution, for a given 
    source function S = s0 + s1 * tau.
    
    Parameters
    ----------
    u : 1-D array
        Dimensionless wavelength in Doppler width units.
    s0, s1: scalar (astropy intensity units)
        Constants in the definition of source function.
    eta : scalar
        Ratio of line to continuum extinction, alpha_l / alpha_c.
    a: scalar
        Broadening of profile
    u: 1-D array
        Normalised wavelength scale. 
    g_eff: scalar
        Effective Lande factor.
    delta_ratio: scalar
        Ratio of Zeeman broadening to Doppler broadening.
    gamma: scalar
        Inclination angle of magnetic field
    chi: scalar
        Azimuth angle of magnetic field
    """
    phi_0 = voigt(a, u)
    phi_r = voigt(a, u + g_eff * delta_ratio) 
    phi_b = voigt(a, u - g_eff * delta_ratio) 
    psi_0 = faraday(a, u)
    psi_r = faraday(a, u + g_eff * delta_ratio) 
    psi_b = faraday(a, u - g_eff * delta_ratio)
    
    phi_delta = 0.5 * (phi_0 - 0.5 * (phi_b + phi_r))
    phi_I = phi_delta * numpy.sin(gamma)**2 + 0.5 * (phi_b + phi_r)
    phi_Q = phi_delta * numpy.sin(gamma)**2 * numpy.cos(2 * chi)
    phi_U = phi_delta * numpy.sin(gamma)**2 * numpy.sin(2 * chi)
    phi_V = 0.5 * (phi_b - phi_r) * numpy.cos(gamma)
    
    psi_delta = 0.5 * (psi_0 - 0.5 * (psi_b + psi_r))
    psi_Q = psi_delta * numpy.sin(gamma)**2 * numpy.cos(2 * chi)
    psi_U = psi_delta * numpy.sin(gamma)**2 * numpy.sin(2 * chi)
    psi_V = 0.5 * (psi_b - psi_r) * numpy.cos(gamma)
    
    kI = 1 + eta * phi_I
    kQ = eta * phi_Q
    kU = eta * phi_U
    kV = eta * phi_V

    fQ = eta * psi_Q
    fU = eta * psi_U
    fV = eta * psi_V

    delta = (kI**4 + kI**2 * (fQ**2 + fU**2 + fV**2 - kQ**2 - kU**2 - kV**2) - 
             (kQ * fQ + kU * fU + kV * fV)**2)
    I = s0 + s1 / delta * kI * (kI**2 + fQ**2 + fU**2 + fV**2)
    Q = -s1 / delta * (kI**2 * kQ - kI * (kU * fV - kV * fU) + fQ * (kQ * fQ + kU * fU + kV * fV))
    U = -s1 / delta * (kI**2 * kU - kI * (kV * fQ - kQ * fV) + fU * (kQ * fQ + kU * fU + kV * fV))
    V = -s1 / delta * (kI**2 * kV + fV * (kQ * fQ + kU * fU + kV * fV))
    return I, Q, U, V


class Stokes_UR():
    '''
    Displays a widget Stokes profiles from the Unno-Rachkovsky solution.
    
    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    '''
    
    """
    Parameters
    ----------
    s0, s1: scalar (astropy intensity units)
        Constants in the definition of source function.
    eta : scalar
        Ratio of line to continuum extinction, alpha_l / alpha_c.
    a: scalar
        Broadening of profile
    u: 1-D array
        Normalised wavelength scale. 
    g_eff: scalar
        Effective Lande factor.
    delta_ratio: scalar
        Ratio of Zeeman broadening to Doppler broadening.
    gamma: scalar
        Inclination angle of magnetic field
    chi: scalar
        Azimuth angle of magnetic field
    """
    # initial parameters
    s0 = 1
    s1 = 5
    npts = 121
    wmax = 10
    log_a = -1.3
    eta = 20
    g_eff = 1.
    delta_ratio = 1.5
    gamma = numpy.pi/3
    chi = 0
        
    def __init__(self):
        self._compute_profile()
        self._make_plot()
        self._make_widget()
    
    def _compute_profile(self):
        """
        Calculates the line profile given a a damping parameter, 
        source function, opacities, and mu.
        """
        self.u = numpy.linspace(-self.wmax, self.wmax, self.npts)
        a = 10. ** self.log_a
        self.I, self.Q, self.U, self.V = unno_rachkovsky(
            self.u, s0=self.s0, s1=self.s1, eta=self.eta, a=a, g_eff=self.g_eff,
            delta_ratio=self.delta_ratio, gamma=self.gamma, chi=self.chi)    

    def _make_plot(self):
        plt.close(1)
        margin = {'top': 25, 'bottom': 35, 'left': 35, 'right':25}
        fig_layout = {'height': '100%', 'width': '100%'}
        self.I_fig = plt.figure(1, title='Stokes I', fig_margin=margin, layout=fig_layout)
        self.I_plot = plt.plot(self.u, self.I)
        plt.xlabel("Δλ / ΔλD")
        
        plt.close(2)
        self.Q_fig = plt.figure(2, title='Stokes Q', fig_margin=margin, layout=fig_layout)
        self.Q_plot = plt.plot(self.u, self.Q)
        plt.xlabel("Δλ / ΔλD")
        
        plt.close(3)
        self.U_fig = plt.figure(3, title='Stokes U', fig_margin=margin, layout=fig_layout)
        self.U_plot = plt.plot(self.u, self.U)
        plt.xlabel("Δλ / ΔλD")
        
        plt.close(4)
        self.V_fig = plt.figure(4, title='Stokes V', fig_margin=margin, layout=fig_layout)
        self.V_plot = plt.plot(self.u, self.V)
        plt.xlabel("Δλ / ΔλD")

        
    def _update_plot(self, log_a, eta, g_eff, delta_ratio, gamma, chi):  
        self.log_a = log_a
        self.eta = eta
        self.g_eff = g_eff
        self.delta_ratio = delta_ratio
        self.gamma = gamma * 2 * numpy.pi / 360
        self.chi = chi * 2 * numpy.pi / 360
        self._compute_profile()
        self.I_plot.y = self.I
        self.Q_plot.y = self.Q
        self.U_plot.y = self.U
        self.V_plot.y = self.V
        
    def _make_widget(self):
        fig = VBox([HBox([self.I_fig, self.Q_fig], layout=Layout(align_items='stretch', height='300px')), 
                    HBox([self.U_fig, self.V_fig], layout=Layout(height='300px'))])
        log_a_slider = FloatSlider(min=-3, max=1., step=0.01, value=self.log_a, description='lg(a)')
        eta_slider = FloatSlider(min=1, max=300., step=0.01, value=self.eta, description='r$\eta$')
        g_eff_slider = FloatSlider(min=-2.5, max=2.5, step=0.1, value=self.g_eff, description='$g_{eff}$')
        delta_ratio_slider = FloatSlider(min=0.0, max=10, step=0.02, value=self.delta_ratio, 
                                         description=r'$\Delta \lambda_B / \Delta \lambda_D$')
        gamma_slider = FloatSlider(min=0, max=180, step=0.05, value=self.gamma * 360 / (2 * numpy.pi), description=r'$\gamma$ ($^o$)')
        chi_slider = FloatSlider(min=0, max=360, step=0.05, value=self.chi * 360 / (2 * numpy.pi), description=r'$\chi$ ($^o$)')

        
        w = interactive(self._update_plot, log_a=log_a_slider, eta=eta_slider, g_eff=g_eff_slider,
                        delta_ratio=delta_ratio_slider, gamma=gamma_slider, chi=chi_slider)
        
        controls = HBox([VBox([w.children[0], w.children[1]]), 
                         VBox([w.children[3], w.children[4]]), 
                         VBox([w.children[2], w.children[5]])])
        self.widget = VBox([controls, fig])
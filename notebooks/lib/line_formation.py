import os
import numpy as np
import bqplot.pyplot as plt
from bqplot import LinearScale, LogScale
from ipywidgets import (interactive, Layout, HBox, VBox, Box, Label,
                        IntSlider, FloatSlider, HTMLMath, Dropdown,
                        )
from scipy.special import wofz
from scipy import interpolate as interp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def voigt(damping, u):
    """
    Calculates the Voigt function.
    """
    z = (u + 1j * damping)
    return wofz(z).real


class Transp():
    '''
    Displays a widget illustrating line formation in a one dimensional model atmosphere.
    Based on XTRANSP.PRO by Mats Carlsson and Oivind Wikstol.
    
    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    '''
    DATA_FILE = os.path.join('data', 'source_functions.npz')
    data = np.load(DATA_FILE)
    # variable names inside data structure
    SFUNCTIONS = {"VAL3C Mg": "s_nu_mg", "VAL3C Ca": "s_nu_ca",
                  "VAL3C LTE": "s_nu_lte"}
    TAUS =  {"VAL3C Mg": "t_500_mg", "VAL3C Ca": "t_500_ca",
             "VAL3C LTE": "t_500_lte"}
    # initial parameters
    mu = 1.0
    npts = 101
    xmax = 50
    a = -2.5
    opa_cont = 0.
    opa_line = 6.44
    source = "VAL3C Mg"
    
    def __init__(self):
        self._compute_profile()
        self._make_plot()
        self._make_widget()
    
    def _compute_profile(self):
        """
        Calculates the line profile given a a damping parameter, 
        source function, opacities, and mu.
        """
        self.tau500 = self.data[self.TAUS[self.source]]
        self.source_function = self.data[self.SFUNCTIONS[self.source]]
        tau500 = self.tau500
        source_function = self.source_function 
        self.freq = np.linspace(-float(self.xmax), self.xmax, self.npts)
        a = 10. ** self.a
        self.h = voigt(a, self.freq)
        self.xq = self.h * 10. ** self.opa_line + 10. ** self.opa_cont
        xq = self.xq
        self.tau500_cont = self.mu / 10 ** self.opa_cont
        self.tau500_line = self.mu / self.xq.max()
        f = interp.interp1d(tau500, source_function, bounds_error=False)
        self.source_function_cont = f(self.tau500_cont)[()]
        self.source_function_line = f(self.tau500_line)[()]
        xq = xq[:, np.newaxis]
        tmp = source_function * np.exp(-xq * tau500 / self.mu) * xq * tau500
        self.prof = np.log(10) / self.mu * np.trapz(tmp.T, np.log(tau500),
                                                    axis=0)

    def _make_plot(self):
        plt.close(1)
        margin = {'top': 25, 'bottom': 35, 'left': 35, 'right':25}
        fig_layout = {'height': '100%', 'width': '100%'}
        self.voigt_fig = plt.figure(1, title='Voigt profile', fig_margin=margin, layout=fig_layout)
        self.voigt_plot = plt.plot(self.freq, self.h, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")
        
        plt.close(2)
        self.abs_fig = plt.figure(2, title='(αᶜ + αˡ) / α₅₀₀', fig_margin=margin, layout=fig_layout)
        self.abs_plot = plt.plot(self.freq, self.xq, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")
        
        plt.close(3)
        self.int_fig = plt.figure(3, title='Intensity', fig_margin=margin, layout=fig_layout)
        self.int_plot = plt.plot(self.freq, self.prof, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")
        
        plt.close(4)
        self.source_fig = plt.figure(4, title='Source Function', fig_margin=margin, layout=fig_layout)
        self.source_plot = plt.plot(np.log10(self.tau500), self.source_function,
                                    scales={'y': LogScale()})
        plt.xlabel("lg(τ₅₀₀)")
        self.tau_labels = plt.label(['τᶜ = 1', 'τˡ = 1'], 
                                    x=np.array([np.log10(self.tau500_cont), np.log10(self.tau500_line)]),
                                    y=np.array([self.source_function_cont, self.source_function_line]),
                                    colors=['black'], y_offset=-25, align='middle')
        self.tau_line_plot = plt.plot(np.array([np.log10(self.tau500_line), np.log10(self.tau500_line)]),
                                      np.array([self.source_function_line / 1.5, self.source_function_line * 1.5]),
                                      colors=['black'])
        self.tau_cont_plot = plt.plot(np.array([np.log10(self.tau500_cont), np.log10(self.tau500_cont)]),
                                      np.array([self.source_function_cont / 1.5, self.source_function_cont * 1.5]),
                                      colors=['black'])
        
    def _update_plot(self, a, opa_cont, opa_line, mu, xmax, source):
        self.a = a
        self.opa_cont = opa_cont
        self.opa_line = opa_line
        self.mu = mu
        self.xmax = xmax
        self.source = source
        self._compute_profile()
        self.voigt_plot.x = self.freq
        self.voigt_plot.y = self.h
        self.abs_plot.x = self.freq
        self.abs_plot.y = self.xq
        self.int_plot.x = self.freq
        self.int_plot.y = self.prof
        self.source_plot.x = np.log10(self.tau500)
        self.source_plot.y = self.source_function
        self.tau_labels.x = np.array([np.log10(self.tau500_cont), np.log10(self.tau500_line)])
        self.tau_labels.y = np.array([self.source_function_cont, self.source_function_line])
        self.tau_line_plot.x = [np.log10(self.tau500_line), np.log10(self.tau500_line)]
        self.tau_line_plot.y = [self.source_function_line / 1.5, self.source_function_line * 1.5]
        self.tau_cont_plot.x = [np.log10(self.tau500_cont), np.log10(self.tau500_cont)]
        self.tau_cont_plot.y = [self.source_function_cont / 1.5, self.source_function_cont * 1.5]
        
    def _make_widget(self):
        fig = VBox([HBox([self.voigt_fig, self.abs_fig], layout=Layout(align_items='stretch', height='300px')), 
                    HBox([self.int_fig, self.source_fig], layout=Layout(height='300px'))])
        a_slider = FloatSlider(min=-5, max=0., step=0.01, value=self.a, description='lg(a)')
        opa_cont_slider = FloatSlider(min=0., max=6., step=0.01, value=self.opa_cont, description=r"$\kappa_c / \kappa_{500}$")
        opa_line_slider = FloatSlider(min=0., max=7., step=0.01, value=self.opa_line, description=r"$\kappa_l / \kappa_{500}$")
        mu_slider = FloatSlider(min=0.01, max=1., step=0.01, value=self.mu, description=r'$\mu$')
        xmax_slider = IntSlider(min=1, max=100, step=1, value=self.xmax, description='xmax')
        source_slider = Dropdown(options=self.SFUNCTIONS.keys(), value=self.source, 
                                 description='Source Function', style={'description_width': 'initial'})
        
        w = interactive(self._update_plot, a=a_slider, opa_cont=opa_cont_slider,
                        opa_line=opa_line_slider, mu=mu_slider, xmax=xmax_slider, source=source_slider)
        
        controls = HBox([VBox([w.children[5], w.children[4]]), 
                         VBox([w.children[0], w.children[3]]), 
                         VBox([w.children[2], w.children[1]])])
        self.widget = VBox([controls, fig])
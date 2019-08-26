import numpy as np
import bqplot.pyplot as plt
from ipywidgets import interactive, Layout, HBox, VBox, Box, Label, IntSlider, FloatSlider, HTMLMath

def slab():
    '''
    Displays a widget illustrating line formation in a homogenous slab.
    
    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    '''
    def _compute_slab(i0, source, tau_cont, tau_line):
        '''
        Calculates slab line profile.
        '''
        NPT = 101
        MAX_DX = 5.
        x = np.arange(NPT) - (NPT - 1.) / 2
        x *= MAX_DX / x.max()
        tau = tau_cont + tau_line * np.exp(-x * x)
        extinc = np.exp(-tau)
        intensity = float(i0) * extinc + float(source) * (1. - extinc)
        return (x, intensity)
    
    I0 = 15
    S = 65
    x, y = _compute_slab(I0, S, 0.5, 0.9)
    base = np.zeros_like(x)
    fig = plt.figure(title='Slab line formation')
    int_plot = plt.plot(x, y, 'b-')
    source_line = plt.plot(x, base + S, 'k--')
    i0_line = plt.plot(x, base + I0, 'k:')
    labels = plt.label(['I₀', 'I', 'S'], 
                       x=np.array([int_plot.x[0] + 0.2, int_plot.x[-1] - 0.2, int_plot.x[0] + 0.2]),
                       y=np.array([i0_line.y[0], int_plot.y[0], source_line.y[0]]) + 2,
                       colors=['black'])
    plt.ylim(0, 100)
    i0_slider = IntSlider(min=0, max=100, value=I0, description=r'$I_0$')
    s_slider = IntSlider(min=0, max=100, value=S, description=r'$S$')
    tau_c_slider = FloatSlider(min=0, max=1., step=0.01, value=0.5, description=r'$\tau_{\mathrm{cont}}$')
    tau_l_slider = FloatSlider(min=0, max=10., step=0.01, value=0.9, description=r'$\tau_{\mathrm{line}}$')

    def plot_update(i0=I0, source=S, tau_cont=0.5, tau_line=0.9):
        x, y = _compute_slab(i0, source, tau_cont, tau_line)
        int_plot.y = y
        source_line.y = base + source
        i0_line.y = base + i0
        labels.y = np.array([i0, y[0], source]) + 2

    widg = interactive(plot_update, i0=i0_slider, source=s_slider, tau_cont=tau_c_slider,
              tau_line=tau_l_slider)
    help_w = HTMLMath("<p><b>Purpose: </b>"
                      "This widget-based procedure is used for "
                      "studying spectral line formation in a "
                      "homogeneous slab.</p>"
                      "<p><b>Inputs:</b></p>"
                      "<ul>"
                     r"   <li>$I_0$: The incident intensity.</li>"
                     r"   <li>$S$: The source function.</li>"
                     r"   <li>$\tau_{\mathrm{cont}}$ : The continuum optical depth.</li>"
                     r"   <li>$\tau_{\mathrm{line}}$ : The integrated optical depth in the spectral line.</li>"
                     "</ul>")

    return HBox([VBox([widg, help_w], layout=Layout(width='33%', top='50px', left='5px')),
                 Box([fig], layout=Layout(width='66%'))],
                 layout=Layout(border='50px'))

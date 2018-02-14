import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fmin
from scipy.stats import linregress
from matplotlib2tikz import save as tikz_save

plt.style.use('bmh')

# Raw data

time = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270,
                 300, 330, 360, 390, 420, 450, 480, 510, 540,
                 570, 600, 630, 660, 690, 720, 750, 780, 810,
                 840, 870, 900, 930, 960, 990, 1020, 1050, 1080,
                 1110, 1140, 1170, 1200, 1230, 1260, 1290,
                 1320, 1350, 1380, 1410, 1440, 1470, 1500])

volume = np.array([0, 13.2522, 22.8151, 35.4172, 48.2249,
                   63.3103, 73.558, 86.1327, 96.7225,
                   107.9742, 118.6576, 128.7267, 140.6426,
                   152.4861, 162.8935, 174.1857, 185.7268,
                   197.1192, 209.5242, 221.4678, 232.2868,
                   247.6275, 259.4228, 270.9724, 281.8857,
                   293.4212, 304.6134, 315.6425, 326.8809,
                   338.671, 348.127, 363.4559, 374.8323,
                   385.8517, 396.7044, 408.1897, 420.0901,
                   427.0895, 437.5906, 448.138, 459.2603,
                   469.2068, 480.2171, 491.9888, 502.4742,
                   513.0803, 524.2078, 535.2854, 545.8359,
                   555.0863, 564.3464])

concentration = np.array([3.023798609, 3.114830065, 3.193132997,
                          3.344098032, 3.545952177, 3.71131258,
                          3.880179942, 4.051567078, 4.145738602,
                          4.204649925, 4.310641861, 4.373957515,
                          4.449874592, 4.523475265, 4.557658434,
                          4.621247387, 4.665646744, 4.688958049,
                          4.724696445, 4.763434982, 4.777761936,
                          4.817210579, 4.840685558, 4.846937061,
                          4.861563492, 4.862069321, 4.865633011,
                          4.883685875, 4.888158989, 4.893807769,
                          4.89773283, 4.902150154, 4.908982992,
                          4.91435442, 4.919867516, 4.916160679,
                          4.928572059, 4.917313385, 4.922553825,
                          4.919561267, 4.92966497, 4.934463024,
                          4.935445404, 4.937192059, 4.930878401,
                          4.932299042, 4.933749437, 4.929975319,
                          4.935555362, 4.937137485, 4.940053844])

def rates_ba(state, t, paras):
    c, q = state

    k1 = paras[0]
    qmax = paras[1]

    dcdt = Vdot / V * (c_input - c) - k1 * c * (qmax - q)
    dqdt = k1 * c * (qmax - q)
    return [dcdt, dqdt]


def conc_ba(t, initial_state, paras):
    x = odeint(rates_ba, initial_state, t, args=(paras,))
    return x


def residual_sq_ba(paras, t, initial_state, data):
    model = conc_ba(t, initial_state, paras)
    c_model = model[:, 0]
    e = ((c_model - data) ** 2).sum()
    return e


def rates_part2_ba(state, t, paras):
    c1, q1, c2, q2 = state

    k1 = paras[0]
    qmax = paras[1]

    dc1dt = Vdot / V * (c_input - c1) - k1 * c1 * (qmax - q1)
    dq1dt = k1 * c1 * (qmax - q1)

    dc2dt = Vdot / V * (c1 - c2) - k1 * c2 * (qmax - q2)
    dq2dt = k1 * c2 * (qmax - q2)

    return [dc1dt, dq1dt, dc2dt, dq2dt]


def conc_part2_ba(t, initial_state, paras):
    x = odeint(rates_part2_ba, initial_state, t, args=(paras,))
    return x


def rates_t(state, t, paras):
    c, q = state

    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]

    dcdt = Vdot / V * (c_input - c) - k1 * c * (qmax - q) + k2 * q
    dqdt = k1 * c * (qmax - q) - k2 * q
    return [dcdt, dqdt]


def conc_t(t, initial_state, paras):
    x = odeint(rates_t, initial_state, t, args=(paras,))
    return x


def residual_sq_t(paras, t, initial_state, data):
    model = conc_t(t, initial_state, paras)
    c_model = model[:, 0]
    e = ((c_model - data) ** 2).sum()
    return e


def rates_part2_t(state, t, paras):
    c1, q1, c2, q2 = state

    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]

    dc1dt = Vdot / V * (c_input - c1) - k1 * c1 * (qmax - q1) + k2 * q1
    dq1dt = k1 * c1 * (qmax - q1) - k2 * q1

    dc2dt = Vdot / V * (c1 - c2) - k1 * c2 * (qmax - q2) + k2 * q2
    dq2dt = k1 * c2 * (qmax - q2) - k2 * q2

    return [dc1dt, dq1dt, dc2dt, dq2dt]


def conc_part2_t(t, initial_state, paras):
    x = odeint(rates_part2_t, initial_state, t, args=(paras,))
    return x

V = 670
Vdot = linregress(time, volume).slope
c_input = 4.94
c_init = concentration[0]
q_init = c_input - c_init
init_state = c_init, q_init
init_state_part2 = c_init, q_init, c_init, 0
t_sim = np.arange(0, 6000, 1)
v_sim = t_sim * Vdot


p_guess_ba = [0.01, 20]
p_guess_t = [0.01, 0.01, 20]

k1fit_ba, qmaxfit_ba = fmin(residual_sq_ba, p_guess_ba, args=(time, init_state, concentration))
k1fit_t, k2fit_t, qmaxfit_t = fmin(residual_sq_t, p_guess_t, args=(time, init_state, concentration))

part2_ba = conc_part2_ba(t_sim, init_state_part2, [k1fit_ba, qmaxfit_ba])
part2_t = conc_part2_t(t_sim, init_state_part2, [k1fit_t, k2fit_t, qmaxfit_t])

plt.gcf().clear()
plt.plot(time, concentration, marker='o', linestyle='None', color='k',markerfacecolor='None', label='Experiment')
plt.plot(time, conc_ba(time, init_state, [k1fit_ba, qmaxfit_ba])[:, 0], linestyle='dashed', label='Bohart-Adams fit')
plt.plot(time, conc_t(time, init_state, [k1fit_t, k2fit_t, qmaxfit_t])[:, 0], linestyle='dashed', label='Thomas Fit')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mg/mL)')
plt.legend(loc='lower right')
tikz_save('comparison_fit.tikz', figureheight='8cm', figurewidth='12cm')

plt.gcf().clear()
plt.plot(v_sim, part2_ba[:, 2], label='B-A simulation, filter 2')
plt.plot(v_sim, part2_t[:, 2], label='Thomas simulation, filter 2')
plt.plot(v_sim, np.repeat(4.7, len(v_sim)), color='k', linestyle='dashed')
plt.xlabel('Volume (mL)')
plt.ylabel('Concentration (mg/mL)')
plt.legend(loc='lower right')
tikz_save('comparison_simulation.tikz', figureheight='8cm', figurewidth='12cm')
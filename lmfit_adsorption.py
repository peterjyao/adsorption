import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import linregress
from scipy.optimize import fmin, leastsq, least_squares
import lmfit
from matplotlib2tikz import save as tikz_save

plt.ioff()
plt.style.use('bmh')

exp_data = pd.read_csv('data_onefilter.txt', delimiter='\t', index_col=False)

time = exp_data['time'].values
concentration = exp_data['concentration'].values
volume = exp_data['volume'].values

Vol = 670  # mL
Vdot = linregress(time, volume).slope  # mL/s
c_input = 4.94  # mg/mL
c_init = concentration[0]  # ca. 3 mg/mL
q_init = c_input - c_init


def rates(state, t, paras):
    c, q = state

    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]
    vol = paras[3]

    dcdt = Vdot / vol * (c_input - c) - k1 * c * (qmax - q) + k2 * q
    dqdt = k1 * c * (qmax - q) - k2 * q
    return [dcdt, dqdt]


def conc(t, c0, q0, k1, k2, qmax, v):
    initial_state = (c0, q0)
    params_list = [k1, k2, qmax, v]
    x = odeint(rates, initial_state, t, args=(params_list, ))
    return x


def conc_index(t, c0, q0, k1, k2, qmax, index, v=Vol):
    x = conc(t, c0, q0, k1, k2, qmax, v)
    return x[:, index]


def residual_sq(paras, t, c0, q0, v, data):
    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]

    model = conc(t, c0, q0, k1, k2, qmax, v)
    model_c = model[:, 0]
    e = ((model_c - data) ** 2).sum()
    return e


def residual(paras, t, c0, q0, v, data):
    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]

    model = conc(t, c0, q0, k1, k2, qmax, v)
    model_c = model[:, 0]
    e = model_c - data
    return e


def rates_twofilters(state, t, paras):
    c1, q1, c2, q2 = state

    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]
    vol = paras[3]

    dc1dt = Vdot / vol * (c_input - c1) - k1 * c1 * (qmax - q1) + k2 * q1
    dq1dt = k1 * c1 * (qmax - q1) - k2 * q1

    dc2dt = Vdot / vol * (c1 - c2) - k1 * c2 * (qmax - q2) + k2 * q2
    dq2dt = k1 * c2 * (qmax - q2) - k2 * q2

    return [dc1dt, dq1dt, dc2dt, dq2dt]


def conc_twofilters(t, initial_state, paras):
    x = odeint(rates_twofilters, initial_state, t, args=(paras,))
    return x


c_model = lmfit.Model(conc_index, independent_vars='t')

# c_model = lmfit.Model(conc_index, independent_vars=['t','index'])

params = c_model.make_params()

params['index'].value = 0  # This will return the c values
params['index'].vary = False

params['k1'].value = 0.01

params['k2'].value = 0.01

params['qmax'].value = 1

params['c0'].value = c_init
params['c0'].vary = False

params['q0'].value = q_init
params['q0'].vary = False

params['v'].vary = False

lmfit_results = c_model.fit(concentration, params, t=time)
print(lmfit_results.fit_report())
lmfit_results.plot()
print(lmfit_results.ci_report())

params['k1'].min = 0
params['k2'].min = 0
params['qmax'].min = 0

bound_results = c_model.fit(concentration, params, t=time)
print(bound_results.fit_report())

k1_lmfit = lmfit_results.params['k1'].value
k2_lmfit = lmfit_results.params['k2'].value
qmax_lmfit = lmfit_results.params['qmax'].value

k1_bound = bound_results.params['k1'].value
k2_bound = bound_results.params['k2'].value
qmax_bound = bound_results.params['qmax'].value

k1_fmin, k2_fmin, qmax_fmin = fmin(residual_sq, [0.01, 0.01, 1], args=(time, c_init, q_init, Vol, concentration))
k1_leastsq, k2_leastsq, qmax_leastsq = leastsq(residual, np.array([0.01, 0.01, 1]),
                                               args=(time, c_init, q_init, Vol, concentration))[0]
k1_least_squares, k2_least_squares, qmax_least_squares = least_squares(residual, [0.01, 0.01, 1],
                                                                       bounds=([0, np.inf]),
                                                                       args=(time, c_init, q_init, Vol,
                                                                             concentration)).x

print([k1_lmfit, k2_lmfit, qmax_lmfit], '\n', [k1_bound, k2_bound, qmax_bound], '\n', [k1_fmin, k2_fmin, qmax_fmin])

rsq_lmfit = residual_sq([k1_lmfit, k2_lmfit, qmax_lmfit], time, c_init, q_init, Vol, concentration)
rsq_bound = residual_sq([k1_bound, k2_bound, qmax_bound], time, c_init, q_init, Vol, concentration)
rsq_fmin = residual_sq([k1_fmin, k2_fmin, qmax_fmin], time, c_init, q_init, Vol, concentration)

print([rsq_lmfit, rsq_bound, rsq_fmin])

c_lmfit = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_lmfit, k2=k2_lmfit, qmax=qmax_lmfit, index=0, v=Vol)
q_lmfit = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_lmfit, k2=k2_lmfit, qmax=qmax_lmfit, index=1, v=Vol)

c_bound = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_bound, k2=k2_bound, qmax=qmax_bound, index=0, v=Vol)
q_bound = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_bound, k2=k2_bound, qmax=qmax_bound, index=1, v=Vol)

c_fmin = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_fmin, k2=k2_fmin, qmax=qmax_fmin, index=0, v=Vol)
q_fmin = c_model.eval(t=time, c0=c_init, q0=q_init, k1=k1_fmin, k2=k2_fmin, qmax=qmax_fmin, index=1, v=Vol)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.plot(time, c_lmfit)
ax1.plot(time, concentration, marker='o', markerfacecolor='None', linestyle='None', markersize=4)
ax1.set_title('lmfit')
ax1.set_ylabel('c (mg/mL)')

ax2.plot(time, c_bound)
ax2.plot(time, concentration, marker='o', markerfacecolor='None', linestyle='None', markersize=4)
ax2.set_title('bound')

ax3.plot(time, c_fmin)
ax3.plot(time, concentration, marker='o', markerfacecolor='None', linestyle='None', markersize=4)
ax3.set_title('fmin')

ax4.plot(time, q_lmfit)
ax4.set_ylabel('q (mg/mL)')

ax5.plot(time, q_bound)
ax5.set_xlabel('Time (s)')

ax6.plot(time, q_fmin)

init_state_twofilters = c_init, q_init, c_init, 0

t_sim = np.arange(0, 6000, 5)
v_sim = t_sim * Vdot

twofilters_lmfit = conc_twofilters(t_sim, init_state_twofilters, [k1_lmfit, k2_lmfit, qmax_lmfit, Vol])
twofilters_bound = conc_twofilters(t_sim, init_state_twofilters, [k1_bound, k2_bound, qmax_bound, Vol])
twofilters_fmin = conc_twofilters(t_sim, init_state_twofilters, [k1_fmin, k2_fmin, qmax_fmin, Vol])

fig2, ((ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(2, 3)

ax7.plot(v_sim, twofilters_lmfit[:, 2])
ax7.hlines(4.7, v_sim[0], v_sim[-1], linestyle='dotted')
ax7.set_title('lmfit')
ax7.set_ylabel('c2 (mg/mL)')

ax8.plot(v_sim, twofilters_bound[:, 2])
ax8.hlines(4.7, v_sim[0], v_sim[-1], linestyle='dotted')
ax8.set_title('bound')

ax9.plot(v_sim, twofilters_fmin[:, 2])
ax9.hlines(4.7, v_sim[0], v_sim[-1], linestyle='dotted')
ax9.set_title('fmin')

ax10.plot(v_sim, twofilters_lmfit[:, 3])
ax10.set_ylabel('q2 (mg/mL)')

ax11.plot(v_sim, twofilters_bound[:, 3])
ax11.set_xlabel('Volume (mL)')

ax12.plot(v_sim, twofilters_fmin[:, 3])

#tikz_save('lmfit_notstrict.tikz', figure=fig2, strict=False)

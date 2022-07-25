from pysb import *
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
from sympy import Piecewise
import re

# scRNA-seq data (Julien Sage's lab)
t_cells_tumor = {'Naive_CD4': 270,
                 'Regulatory_CD4': 60,
                 'Exhausted_CD4': 260,
                 'Naive_CD8': 60,
                 'Cytotoxic_CD8': 558,
                 'Exhausted_CD8': 80}

t_cells_spleen = {'Naive_CD4': 599,
                  'Regulatory_CD4': 325,
                  'Exhausted_CD4': 603,
                  'Naive_CD8': 276,
                  'Cytotoxic_CD8': 222,
                  'Exhausted_CD8': 230}

Model()

# Cell types

Monomer('N4')  # naive CD4+ T cell
Monomer('R4', ['c8'])  # regulatory CD4+ T cell
Monomer('E4')  # exhausted CD4+ T cell
Monomer('N8')  # naive CD8+ T cell
Monomer('C8', ['r4'])  # cytotoxic CD8+ T cell
Monomer('E8')  # exhausted CD8+ T cell
Monomer('AgPC', ['state'], {'state': ['a', 'i']})
Monomer('S', ['state'], {'state': ['NE', 'NonNE']})  # SCLC tumor cell
# Monomer('GD')  # gamma-delta T cell

Parameter('N4_0', t_cells_spleen['Naive_CD4'])
Parameter('R4_0', t_cells_spleen['Regulatory_CD4'])
Parameter('E4_0', t_cells_spleen['Exhausted_CD4'])
Parameter('N8_0', t_cells_spleen['Naive_CD8'])
Parameter('C8_0', t_cells_spleen['Cytotoxic_CD8'])
Parameter('E8_0', t_cells_spleen['Exhausted_CD8'])
Parameter('AgPC_0', 100)
Parameter('NE_0', 100)  # 100
Parameter('NonNE_0', 80)  # 50

Initial(N4(), N4_0)
Initial(R4(c8=None), R4_0)
Initial(E4(), E4_0)
Initial(N8(), N8_0)
Initial(C8(r4=None), C8_0)
Initial(E8(), E8_0)
Initial(AgPC(state='i'), AgPC_0)
Initial(S(state='NE'), NE_0)
Initial(S(state='NonNE'), NonNE_0)

# Observables

Observable('Naive_CD4', N4())
Observable('Regulatory_CD4', R4())
Observable('Exhausted_CD4', E4())
Observable('Naive_CD8', N8())
Observable('Cytotoxic_CD8', C8())
Observable('Exhausted_CD8', E8())
Observable('SCLC_cells', S())
Observable('NE_cells', S(state='NE'))
Observable('NonNE_cells', S(state='NonNE'))
Observable('Active_AgPCs', AgPC(state='a'))

# Division and death rules

CC = 1e6  # carrying capacity

Parameter('k_N4_div', 1)
Parameter('k_N4_die', 1)
Rule('naive_CD4_div', N4() >> N4() + N4(), k_N4_div)
Rule('naive_CD4_die', N4() >> None, k_N4_die)

Parameter('k_R4_div', 1)
Parameter('k_R4_die', 1)
Rule('reg_CD4_div', R4(c8=None) >> R4(c8=None) + R4(c8=None), k_R4_div)
Rule('reg_CD4_die', R4(c8=None) >> None, k_R4_die)

Parameter('k_E4_div', 1)
Parameter('k_E4_die', 1)
Rule('exhaust_CD4_div', E4() >> E4() + E4(), k_E4_div)
Rule('exhaust_CD4_die', E4() >> None, k_E4_die)

Parameter('k_N8_div', 1)
Parameter('k_N8_die', 1)
Rule('naive_CD8_div', N8() >> N8() + N8(), k_N8_div)
Rule('naive_CD8_die', N8() >> None, k_N8_die)

Parameter('k_C8_div', 1)
Parameter('k_C8_die', 1)
Rule('cytotox_CD8_div', C8(r4=None) >> C8(r4=None) + C8(r4=None), k_C8_div)
Rule('cytotox_CD8_die', C8(r4=None) >> None, k_C8_die)

Parameter('k_E8_div', 1)
Parameter('k_E8_die', 1)
Rule('exhaust_CD8_div', E8() >> E8() + E8(), k_E8_div)
Rule('exhaust_CD8_die', E8() >> None, k_E8_die)

Parameter('k_NE_div', 10)
Parameter('k_NE_die', 1)
Expression('k_NE_cc', Piecewise((0, NE_cells <= 0),
                                ((k_NE_div-k_NE_die)/CC * SCLC_cells/NE_cells, True)))
Rule('NE_cell_div', S(state='NE') >> S(state='NE') + S(state='NE'), k_NE_div)
Rule('NE_cell_die', S(state='NE') >> None, k_NE_die)
Rule('NE_cell_cc', S(state='NE') + S(state='NE') >> S(state='NE'), k_NE_cc)

Parameter('k_NonNE_div', 10)
Parameter('k_NonNE_die', 1)
Expression('k_NonNE_cc', Piecewise((0, NonNE_cells <= 0),
                                   ((k_NonNE_div-k_NonNE_die)/CC * SCLC_cells/NonNE_cells, True)))
Rule('NonNE_cell_div', S(state='NonNE') >> S(state='NonNE') + S(state='NonNE'), k_NonNE_div)
Rule('NonNE_cell_die', S(state='NonNE') >> None, k_NonNE_die)
Rule('NonNE_cell_cc', S(state='NonNE') + S(state='NonNE') >> S(state='NonNE'), k_NonNE_cc)

# Rule('gamma_delta_div', GD() >> GD() + GD())
# Rule('gamma_delta_die', GD() >> None)

# TODO: Ask about the following three rules (N4 and N8 synthesis and spontaneous APC activation

# Synthesis rules

# Parameter('k_N4_synth', 1)
# Rule('N4_synthesis', None >> N4(), k_N4_synth)

# Parameter('k_N8_synth', 1)
# Rule('N8_synthesis', None >> N8(), k_N8_synth)

# Cell state changes

# AgPC -> AgPC*
# Parameter('k_AgPC_act_basal', 0.1)
# Rule('AgPC_activation_basal', AgPC(state='i') >> AgPC(state='a'), k_AgPC_act_basal)

# AgPC + S -> AgPC* + S
Parameter('k_AgPC_act_tumor', 1)
Rule('AgPC_activation_tumor', AgPC(state='i') + S() >> AgPC(state='a') + S(), k_AgPC_act_tumor)

# AgPC* -> AgPC
Parameter('k_AgPC_deact', 1)
Rule('AgPC_inactivation', AgPC(state='a') >> AgPC(state='i'), k_AgPC_deact)

# N4 + AgPC* -> R4 + AgPC*
Parameter('k_N4_R4_AgPC', 1)
Rule('naive_to_reg_CD4_AgPC', N4() + AgPC(state='a') >> R4(c8=None) + AgPC(state='a'), k_N4_R4_AgPC)

# N4 + AgPC* -> E4 + AgPC*
Parameter('k_N4_E4_AgPC', 1)
Rule('naive_to_exhaust_CD4_AgPC', N4() + AgPC(state='a') >> E4() + AgPC(state='a'), k_N4_E4_AgPC)

# N4 + S -> R4 + S
Parameter('k_N4_C4_S', 1)
Rule('naive_to_reg_CD4_S', N4() + S() >> R4(c8=None) + S(), k_N4_C4_S)

# N8 + AgPC* -> C8 + AgPC*
Parameter('k_N8_C8_AgPC', 1)
Rule('naive_to_cytotox_CD8', N8() + AgPC(state='a') >> C8(r4=None) + AgPC(state='a'), k_N8_C8_AgPC)

# C8 + S -> C8
Parameter('k_S_die_C8', 1)
Rule('S_die_C8', C8(r4=None) + S() >> C8(r4=None), k_S_die_C8)

# C8 + S -> C8 + S + S
Parameter('k_S_div_C8', 1)
Rule('NE_div_C8', C8(r4=None) + S(state='NE') >> C8(r4=None) + S(state='NE') + S(state='NE'), k_S_div_C8)
Rule('NonNE_div_C8', C8(r4=None) + S(state='NonNE') >> C8(r4=None) + S(state='NonNE') + S(state='NonNE'), k_S_div_C8)

# C8 + S -> E8 + S
Parameter('k_C8_E8_S', 1)
Rule('cytotox_to_exhaust_CD8', C8(r4=None) + S() >> E8() + S(), k_C8_E8_S)

# R4 + C8 <-> R4:C8
Parameter('kf_R4_inhibit_C8', 1)
Parameter('kr_R4_inhibit_C8', 1)
Rule('reg_CD4_inhibits_cytotox_CD8', R4(c8=None) + C8(r4=None) | R4(c8=1) % C8(r4=1), kf_R4_inhibit_C8, kr_R4_inhibit_C8)

# Rule('S_div_gamma', S() + GD() >> S() + S() + GD())
# Rule('gamma_delta_inhibits_S_death')

# Run simulations

tspan = np.linspace(0, 5, 101)
sim = ScipyOdeSimulator(model, tspan, verbose=True)
output = sim.run()

for obs in model.observables:
    if re.search('_cells', obs.name) is None:
        plt.figure('tcells')
    else:
        plt.figure('sclc')
    p = plt.plot(tspan, output.observables[obs.name], lw=2, label=obs.name)
    if obs.name in t_cells_tumor.keys():
        plt.plot(tspan[-1], t_cells_tumor[obs.name], '*', ms=10, color=p[-1].get_color())
        #####
        print(obs.name, '%g' % output.observables[obs.name][-1], '(%d)' % t_cells_tumor[obs.name])
        #####
    plt.xlabel('time')
    plt.ylabel('number')
    plt.legend(loc=0)


plt.show()

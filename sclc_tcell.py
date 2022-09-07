from pysb import *
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
from sympy import Piecewise
import re

# scRNA-seq data (Julien Sage's lab)
t_cells_spleen = {'Naive_CD4': 599,
                  'Regulatory_CD4': 325,
                  'Exhausted_CD4': 603,
                  'Naive_CD8': 276,
                  'Cytotoxic_CD8': 222,
                  'Exhausted_CD8': 230}

t_cells_tumor = {'Naive_CD4': 270,
                 'Regulatory_CD4': 60,
                 'Exhausted_CD4': 260,
                 'Naive_CD8': 60,
                 'Cytotoxic_CD8': 558,
                 'Exhausted_CD8': 491}

Model()

# Cell types
Monomer('N4')  # naive CD4+ T cell
Monomer('R4', ['state'], {'state': ['a', 'i']})  # regulatory CD4+ T cell
Monomer('E4')  # exhausted CD4+ T cell
Monomer('N8')  # naive CD8+ T cell
Monomer('C8', ['state'], {'state': ['a', 'i']})  # cytotoxic CD8+ T cell
Monomer('E8')  # exhausted CD8+ T cell
Monomer('AgPC', ['state'], {'state': ['a', 'i']})
Monomer('S', ['state'], {'state': ['NE', 'NonNE']})  # SCLC tumor cell
# Monomer('GD')  # gamma-delta T cell

Parameter('N4_0', 0)  # t_cells_spleen['Naive_CD4'])
Parameter('R4_0', 0)  # t_cells_spleen['Regulatory_CD4'])
Parameter('E4_0', 0)  # t_cells_spleen['Exhausted_CD4'])
Parameter('N8_0', 0)  # t_cells_spleen['Naive_CD8'])
Parameter('C8_0', 0)  # t_cells_spleen['Cytotoxic_CD8'])
Parameter('E8_0', 0)  # t_cells_spleen['Exhausted_CD8'])
Parameter('AgPC_0', 100)
Parameter('NE_0', 0)  # 100
Parameter('NonNE_0', 0)  # 50

Initial(N4(), N4_0)
Initial(R4(state='a'), R4_0)
Initial(E4(), E4_0)
Initial(N8(), N8_0)
Initial(C8(state='a'), C8_0)
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
# Observable('Cytotoxic_CD8_active', C8(state='a'))
Observable('Exhausted_CD8', E8())
Observable('SCLC_cells', S())
Observable('NE_cells', S(state='NE'))
Observable('NonNE_cells', S(state='NonNE'))
# Observable('Active_AgPCs', AgPC(state='a'))

# Division and death rules
# Parameter('k_N4_div', 0)
Parameter('k_N4_die', 9.914)
# Rule('naive_CD4_div', N4() >> N4() + N4(), k_N4_div)
Rule('N4_die', N4() >> None, k_N4_die)

# Parameter('k_R4_div', 0)
Parameter('k_R4_die', 1.633)
# Rule('reg_CD4_div', R4() >> R4() + R4(), k_R4_div)
Rule('R4_die', R4() >> None, k_R4_die)

# Parameter('k_E4_div', 0)
Parameter('k_E4_die', 0.880)
# Rule('exhaust_CD4_div', E4() >> E4() + E4(), k_E4_div)
Rule('E4_die', E4() >> None, k_E4_die)

# Parameter('k_N8_div', 0)
Parameter('k_N8_die', 10)
# Rule('naive_CD8_div', N8() >> N8() + N8(), k_N8_div)
Rule('N8_die', N8() >> None, k_N8_die)

# Parameter('k_C8_div', 0)
Parameter('k_C8_die', 4.198)
# Rule('cytotox_CD8a_div', C8(state='a') >> C8(state='a') + C8(state='a'), k_C8_div)
# Rule('cytotox_CD8i_div', C8(state='i') >> C8(state='i') + C8(state='i'), k_C8_div)
Rule('C8_die', C8() >> None, k_C8_die)

# Parameter('k_E8_div', 0)
Parameter('k_E8_die', 5)
# Rule('exhaust_CD8_div', E8() >> E8() + E8(), k_E8_div)
Rule('E8_die', E8() >> None, k_E8_die)

# Rule('gamma_delta_div', GD() >> GD() + GD())
# Rule('gamma_delta_die', GD() >> None)

# Baseline T cell dynamics

# AgPC <-> AgPC*
Parameter('k_AgPC_act', 1)
Parameter('k_AgPC_deact', 10.286)
Rule('AgPC_activ_basal', AgPC(state='i') | AgPC(state='a'), k_AgPC_act, k_AgPC_deact)

# Naive CD4+ T cell synthesis
Parameter('k_N4_synth', 7000)
Rule('N4_synth', None >> N4(), k_N4_synth)

# N4 + AgPC* -> R4 + AgPC*
Parameter('k_N4_R4_AgPC', 0.1)
Rule('N4_to_R4_AgPC', N4() + AgPC(state='a') >> R4(state='a') + AgPC(state='a'), k_N4_R4_AgPC)

# N4 + AgPC* -> E4 + AgPC*
Parameter('k_N4_E4_AgPC', 0.1)
Rule('N4_to_E4_AgPC', N4() + AgPC(state='a') >> E4() + AgPC(state='a'), k_N4_E4_AgPC)

# Naive CD8+ T cell synthesis
Parameter('k_N8_synth', 5000)
Rule('N8_synth', None >> N8(), k_N8_synth)

# N8 + AgPC* -> C8 + AgPC*
Parameter('k_N8_C8_AgPC', 0.916)
Rule('N8_to_C8_AgPC', N8() + AgPC(state='a') >> C8(state='a') + AgPC(state='a'), k_N8_C8_AgPC)

# C8 -> E8
Parameter('k_C8_E8', 4.790)
Rule('C8_to_E8', C8() >> E8(), k_C8_E8)
# Rule('C8_to_E8_AgPC', C8() + AgPC(state='a') >> E8() + AgPC(state='a'), k_C8_E8)

# Tumor-modified T cell dynamics #####

# R4 + C8-a -> R4 + C8-i
# C8-i -> C8-a
Parameter('k_R4_inhibit_C8', 1)
Parameter('k_uninhibit_C8', 10)
Rule('R4_inhibit_C8', R4(state='a') + C8(state='a') >> R4(state='a') + C8(state='i'), k_R4_inhibit_C8)
Rule('C8_uninhibit', C8(state='i') >> C8(state='a'), k_uninhibit_C8)

# AgPC + S -> AgPC* + S
Parameter('k_AgPC_act_tumor', 1.482e-7)  # 1.094e-4
Rule('AgPC_activ_tumor', AgPC(state='i') + S() >> AgPC(state='a') + S(), k_AgPC_act_tumor)

# C8 + S -> C8
# Parameter('k_S_die_C8', 0.01)
# Rule('S_die_C8', C8(state='a') + S() >> C8(state='a'), k_S_die_C8)

# C8 + S -> C8 + 2S
Parameter('k_NE_div_C8', 0.01)
Rule('NE_div_C8', C8(state='a') + S(state='NE') >>
     C8(state='a') + S(state='NE') + S(state='NE'), k_NE_div_C8)

Parameter('k_NonNE_div_C8', 0.01)
Rule('NonNE_div_C8', C8(state='a') + S(state='NonNE') >>
     C8(state='a') + S(state='NonNE') + S(state='NonNE'), k_NonNE_div_C8)

# S + R4 -> S
Parameter('k_R4_die_S', 1.265e-7)  # 9.340e-5
Rule('R4_die_S', S() + R4() >> S(), k_R4_die_S)

# S + E4 -> S
Parameter('k_E4_die_S', 2.733e-8)  # 2.019e-5
Rule('E4_die_S', S() + E4() >> S(), k_E4_die_S)

# Tumor cell proliferation

CC = 1e9  # carrying capacity

Parameter('k_NE_div', 3)
Parameter('k_NE_die', 1)
# Expression('k_NE_cc', Piecewise((0, NE_cells <= 0),
#                                 ((k_NE_div-k_NE_die)/CC * SCLC_cells/NE_cells, True)))
Expression('k_NE_cc', Piecewise((0, NE_cells <= 0),
                                ((k_NE_div-k_NE_die+k_NE_div_C8*1000)/CC * SCLC_cells/NE_cells, True)))
Rule('NE_cell_div', S(state='NE') >> S(state='NE') + S(state='NE'), k_NE_div)
Rule('NE_cell_die', S(state='NE') >> None, k_NE_die)
Rule('NE_cell_cc', S(state='NE') + S(state='NE') >> S(state='NE'), k_NE_cc)

Parameter('k_NonNE_div', 1)
Parameter('k_NonNE_die', 1)
Expression('k_NonNE_cc', Piecewise((0, NonNE_cells <= 0),
                                   ((k_NonNE_div-k_NonNE_die+k_NonNE_div_C8*1000)/CC * SCLC_cells/NonNE_cells, True)))
Rule('NonNE_cell_div', S(state='NonNE') >> S(state='NonNE') + S(state='NonNE'), k_NonNE_div)
Rule('NonNE_cell_die', S(state='NonNE') >> None, k_NonNE_die)
Rule('NonNE_cell_cc', S(state='NonNE') + S(state='NonNE') >> S(state='NonNE'), k_NonNE_cc)

# T cell antibody treatments #########

Monomer('AntiCD4')
Parameter('AntiCD4_0')
Initial(AntiCD4(), AntiCD4_0)
Parameter('k_antiCD4_inhibit_R4', 1)
Parameter('k_uninhibit_R4', 10)
Rule('AntiCD4_inhibit_R4', AntiCD4() + R4(state='a') >> AntiCD4() + R4(state='i'), k_antiCD4_inhibit_R4)
Rule('R4_uninhibit', R4(state='i') >> R4(state='a'), k_uninhibit_R4)

Monomer('AntiCD8')
Parameter('AntiCD8_0')
Initial(AntiCD8(), AntiCD8_0)
Parameter('k_antiCD8_inhibit_C8', 1)
Rule('AntiCD8_inhibit_C8', AntiCD8() + C8(state='a') >> AntiCD8() + C8(state='i'), k_antiCD8_inhibit_C8)

# run simulations

# equilibrate systems without tumor cells
tspan = np.linspace(0, 6, 1001)  # 0, 20, 1001)
sim = ScipyOdeSimulator(model, tspan, verbose=True)
output = sim.run()

# for i, sp in enumerate(model.species):
#     print(i, sp)
# quit()

# add tumor cells
initials = output.species[-1]
idx = [str(sp) for sp in model.species].index('S(state=\'NE\')')
initials[idx] = 100
output = sim.run(initials=initials)

# add anti-CD4 and/or anti-CD8 antibodies #####
ADD_ANTI_CD4 = True
ADD_ANTI_CD8 = False

if ADD_ANTI_CD4:
    idx = [str(sp) for sp in model.species].index('AntiCD4()')
    initials[idx] = 10

if ADD_ANTI_CD8:
    idx = [str(sp) for sp in model.species].index('AntiCD8()')
    initials[idx] = 1e5

output2 = sim.run(initials=initials)

#####
t_cells = t_cells_tumor
# t_cells = t_cells_spleen
#####

# print('Active_AgPCs', output.observables['Active_AgPCs'][-1])
# print('SCLC_cells', output.observables['SCLC_cells'][-1])
# print('Cytotoxic_CD8_active', output.observables['Cytotoxic_CD8_active'][-1])
print()

for obs in model.observables:
    if obs.name == 'NE_cells' or obs.name == 'NonNE_cells':
        continue
    if re.search('_cells', obs.name) is None:
        plt.figure('tcells')
    else:
        plt.figure('sclc')
    label = 'SCLC cells' if obs.name == 'SCLC_cells' else '%s%s' % (obs.name[0], obs.name[-1])
    p = plt.plot(tspan, output.observables[obs.name], lw=3, label=label)
    #####
    if ADD_ANTI_CD4 or ADD_ANTI_CD8:
        plt.plot(tspan, output2.observables[obs.name], '--', lw=3, color=p[-1].get_color())
    #####
    if obs.name in t_cells.keys():
        plt.plot(tspan[0], t_cells_spleen[obs.name], '*', ms=10, color=p[-1].get_color())
        plt.plot(tspan[-1], t_cells[obs.name], '*', ms=10, color=p[-1].get_color())
        #####
        print(obs.name, '%g' % output.observables[obs.name][-1], '(%d)' % t_cells[obs.name])
        #####
    plt.xlabel('time', fontsize=20)
    plt.ylabel('number', fontsize=20)
    # plt.xticks([0, 5, 10, 15, 20], fontsize=20)
    plt.xticks([3, 4, 5, 6], ['0', '1', '2', '3'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(left=3)
    plt.legend(loc=0, fontsize=18)

plt.figure('tcells')
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 1), ncol=2, fontsize=18, columnspacing=0.5)
plt.tight_layout()
plt.savefig('tcells.pdf', format='pdf')

plt.figure('sclc')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.gca().yaxis.get_offset_text().set_fontsize(20)
# plt.yticks([0, 1e5, 2e5, 3e5, 4e5])
plt.tight_layout()
plt.savefig('sclc.pdf', format='pdf')

plt.show()

# ~~~
# This file is part of the paper:
#
#   "Model Reduction for Large Scale Systems"
#
#   https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt
#
# Copyright 2019-2021 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2021)
# ~~~

import matplotlib.pyplot as plt
# import tikzplotlib
import sys
import numpy as np
path = '../../../'
sys.path.append(path)
from pdeopt.tools import get_data

directory = 'Starter9/'

mu_error = True

colorclass0 =(0.65, 0.00, 0.15)
colorclass1 =(0.84, 0.19, 0.15)
colorclass2 =(0.99, 0.68, 0.38)
colorclass3 =(0.96, 0.43, 0.26)
colorclass4 = 'black'
colorclass5 =(0.17, 0.25, 0.98)

# I want to have these methods in my plot: 
method_tuple = [
                [1, 'FOM BFGS'],
                [8, 'BFGS NCD TR-RB Keil et al. 2020'],
                [10, 'BFGS PG TR-RB']
                ]

times_full_0 , J_error_0, mu_error_0, FOC_0, _ = get_data(directory,method_tuple[0][0], FOC=True, j_list=False)
times_full_1 , J_error_1, mu_error_1, FOC_1, j_list_1 = get_data(directory,method_tuple[1][0], FOC=True)
times_full_2 , J_error_2, mu_error_2, FOC_2, j_list_2 = get_data(directory,method_tuple[2][0], FOC=True)

if 1:
    timings_figure = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0 ,J_error_0 , '-', color=colorclass0, marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_1 ,J_error_1 , '-', color=colorclass1, marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2 ,J_error_2 , '-', color=colorclass2, marker='x', label=method_tuple[2][1])
    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('$| \hat{\mathcal{J}}_h(\overline{\mu})-\hat{\mathcal{J}}^k_n(\mu_k) |$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    plt.legend(fontsize=10)
    # plt.legend(loc='lower center', fontsize=10)

    # tikzplotlib.save("{}J_error.tex".format(directory))
    # timings_figure.savefig('{}J_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

if 1:
    timings_figure_3 = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0, FOC_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_1, FOC_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2, FOC_2 , '-', color=colorclass2 , marker='o', label=method_tuple[2][1])
    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('FOC condition', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    # plt.legend(loc='lower center', fontsize=10)
    plt.legend(fontsize=10)

    # tikzplotlib.save("{}FOC.tex".format(directory))
    # timings_figure_3.savefig('{}FOC.pdf'.format(directory), format='pdf', bbox_inches="tight")

if mu_error is True:
    timings_figure = plt.figure(figsize=(10,10))
    # plt.semilogy(times_full_0 ,mu_error_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_1 ,mu_error_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2 ,mu_error_2 , '-', color=colorclass2 , marker='x', label=method_tuple[2][1])
    plt.xlabel('Time [s]',fontsize=28)
    #plt.ylabel('$\| \overline{\mu}-\mu_k \|$', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.xlim((0,300))
    # plt.ylim((1e-7,1e5))
    plt.grid()
    plt.legend(fontsize=30, bbox_to_anchor= [1.025,1.025], loc= 'upper right')

    # tikzplotlib.save("{}mu_error.tex".format(directory))
    # timings_figure.savefig('{}mu_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

plt.show()

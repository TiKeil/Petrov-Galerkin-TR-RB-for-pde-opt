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
#   Luca Mechelli (2020)
#   Tim Keil      (2019 - 2021)
# ~~~

import numpy as np


def plot_functional(opt_fom, steps, ranges):
    first_component_steps = steps
    second_component_steps = steps
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    mu_first_component = np.linspace(ranges[0][0],ranges[0][1],first_component_steps)
    mu_second_component = np.linspace(ranges[1][0],ranges[1][1],second_component_steps)

    x1,y1 = np.meshgrid(mu_first_component,mu_second_component)
    func_ = np.zeros([second_component_steps,first_component_steps]) #meshgrid shape the first component as column index

    for i in range(first_component_steps):
        for j in range(second_component_steps):
            mu_ = opt_fom.parameters.parse([x1[j][i],y1[j][i]])
            func_[j][i] = opt_fom.output_functional_hat(mu_)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1, y1, func_, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    #fig.savefig('3d', format='pdf', bbox_inches="tight")

    fig2 = plt.figure()

    number_of_contour_levels= 100
    cont = plt.contour(x1,y1,func_,number_of_contour_levels)

    fig2.colorbar(cont, shrink=0.5, aspect=5)
    # fig2.savefig('2d', format='pdf', bbox_inches="tight")
    return x1, y1, func_


def compute_errors(opt_fom, parameter_space, J_start, J_opt,
                   mu_start, mu_opt, mus, Js, times, tictoc, FOC):
    mu_error = [np.linalg.norm(mu_start.to_numpy() - mu_opt)]
    J_error = [J_start - J_opt]
    for mu_i in mus[1:]: # the first entry is mu_start
        if isinstance(mu_i,dict):
            mu_error.append(np.linalg.norm(mu_i.to_numpy() - mu_opt))
        else:
            mu_error.append(np.linalg.norm(mu_i - mu_opt))

    i = 1 if (len(Js) >= len(mus)) else 0
    for Ji in Js[i:]: # the first entry is J_start
        J_error.append(np.abs(Ji - J_opt))
    times_full = [tictoc]
    for tim in times:
        times_full.append(tim + tictoc)

    if len(FOC)!= len(times_full):
        print("Computing only the initial FOC")
        gradient = opt_fom.output_functional_hat_gradient(mu_start)
        mu_box = opt_fom.parameters.parse(mu_start.to_numpy()-gradient)
        from pdeopt.TR import projection_onto_range
        first_order_criticity = mu_start.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)
        FOCs= [normgrad]
        FOCs.extend(FOC)
    else:
        FOCs = FOC

    if len(J_error) > len(times_full):
        # this happens sometimes in the optional enrichment. For this we need to compute the last J error
        # the last entry is zero and only there to detect this case
        assert not Js[-1]
        J_error.pop(-1)
        J_error.pop(-1)
        J_error.append(np.abs(J_opt-Js[-2]))
    return times_full, J_error, mu_error, FOCs


import scipy
def compute_eigvals(A,B):
    print('WARNING: THIS MIGHT BE VERY EXPENSIVE')
    return scipy.sparse.linalg.eigsh(A, M=B, return_eigenvectors=False)

import csv
def save_data(directory, times, J_error, n, mu_error=None, FOC=None, additional_data=None):
    with open('{}/error_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in J_error:
            writer.writerow([val])
    with open('{}/times_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in times:
            writer.writerow([val])
    if mu_error is not None:
        with open('{}/mu_error_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in mu_error:
                writer.writerow([val])
    if FOC is not None:
        with open('{}/FOC_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in FOC:
                writer.writerow([val])
    if additional_data:
        for key in additional_data.keys():
            if not key == "opt_rom":
                with open('{}/{}_{}.txt'.format(directory, key, n), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in additional_data[key]:
                        writer.writerow([val])


def get_data(directory, n, mu_error_=True, mu_est_=False, FOC=False, j_list=True):
    J_error = []
    times = []
    mu_error = []
    mu_time = []
    mu_est = []
    FOC_ = []
    j_list_ = []
    if mu_error_ is True:
        f = open('{}mu_error_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_error.append(float(val[0]))
    if FOC is True:
        f = open('{}FOC_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            FOC_.append(float(val[0]))
    if j_list is True:
        f = open('{}j_list_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            j_list_.append(float(val[0]))
    if mu_est_ is True:
        f = open('{}mu_est_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_est.append(float(val[0]))
        f = open('{}mu_time_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_time.append(float(val[0]))
    f = open('{}error_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        J_error.append(abs(float(val[0])))
    f = open('{}times_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        times.append(float(val[0]))
    if mu_error_:
        if mu_est_:
            if FOC:
                return times, J_error, mu_error, mu_time, mu_est, FOC_, j_list_
            else:
                return times, J_error, mu_error, mu_time, mu_est, j_list_
        else:
            if FOC:
                return times, J_error, mu_error, FOC_, j_list_
            else:
                return times, J_error, mu_error, j_list_
    else:
        if FOC:
            return times, J_error, FOC_, j_list_
        else:
            return times, J_error, j_list


def compute_all_errors_and_estimators_for_all_ROMS(validation_set, opt_fom, opt_roms_1a, opt_roms_2a, opt_roms_3a,
                                                   opt_roms_4a, reductor_4a, reductor_5a):
    J_errors_1as, rel_J_errors_1as, J_estimators_1as, effectivities_J_1as = [], [], [], []
    J_errors_2as, rel_J_errors_2as, J_estimators_2as, effectivities_J_2as = [], [], [], []
    J_errors_3as, rel_J_errors_3as, J_estimators_3as, effectivities_J_3as = [], [], [], []
    J_errors_4as, rel_J_errors_4as, J_estimators_4as, effectivities_J_4as = [], [], [], []

    DJ_errors_1as, rel_DJ_errors_1as = [], []
    DJ_errors_2as, rel_DJ_errors_2as = [], []
    DJ_errors_3as, rel_DJ_errors_3as = [], []
    DJ_errors_4as, rel_DJ_errors_4as = [], []

    u_errors_4as, rel_u_errors_4as, u_estimators_4as, effectivities_u_4as = [], [], [], []
    u_errors_5as, rel_u_errors_5as, u_estimators_5as, effectivities_u_5as = [], [], [], []
    p_errors_4as, rel_p_errors_4as, p_estimators_4as, effectivities_p_4as = [], [], [], []
    p_errors_5as, rel_p_errors_5as, p_estimators_5as, effectivities_p_5as = [], [], [], []

    Js = []

    u_hs, p_hs, J_hs, DJ_hs = [], [], [], []
    print('computing expensive FOM parts')
    for mu in validation_set:
        print('.', end='', flush=True)
        u_h = opt_fom.solve(mu)
        p_h = opt_fom.solve_dual(mu, U=u_h)
        actual_J = opt_fom.output_functional_hat(mu, U=u_h, P=p_h)
        actual_DJ = opt_fom.output_functional_hat_gradient(mu, U=u_h, P=p_h)
        u_hs.append(u_h)
        p_hs.append(p_h)
        J_hs.append(actual_J)
        DJ_hs.append(actual_DJ)

    print('\ncomputing the rest')
    for opt_rom_1a, opt_rom_2a, opt_rom_3a, opt_rom_4a in zip(opt_roms_1a, opt_roms_2a, opt_roms_3a, opt_roms_4a):
        opt_rom_5a = opt_rom_2a
        J_errors_1a, rel_J_errors_1a, J_estimators_1a, effectivities_J_1a = [], [], [], []
        J_errors_2a, rel_J_errors_2a, J_estimators_2a, effectivities_J_2a = [], [], [], []
        J_errors_3a, rel_J_errors_3a, J_estimators_3a, effectivities_J_3a = [], [], [], []
        J_errors_4a, rel_J_errors_4a, J_estimators_4a, effectivities_J_4a = [], [], [], []

        DJ_errors_1a, rel_DJ_errors_1a = [], []
        DJ_errors_2a, rel_DJ_errors_2a = [], []
        DJ_errors_3a, rel_DJ_errors_3a = [], []
        DJ_errors_4a, rel_DJ_errors_4a = [], []

        u_errors_4a, rel_u_errors_4a, u_estimators_4a, effectivities_u_4a = [], [], [], []
        u_errors_5a, rel_u_errors_5a, u_estimators_5a, effectivities_u_5a = [], [], [], []
        p_errors_4a, rel_p_errors_4a, p_estimators_4a, effectivities_p_4a = [], [], [], []
        p_errors_5a, rel_p_errors_5a, p_estimators_5a, effectivities_p_5a = [], [], [], []

        J = []
        for i, mu in enumerate(validation_set):
            print('.', end='', flush=True)
            u_h = u_hs[i]
            p_h = p_hs[i]
            actual_J = J_hs[i]
            actual_DJ = DJ_hs[i]
            J.append(actual_J)

            # this is equivalent for 1 and 2
            U = opt_rom_1a.solve(mu)
            P = opt_rom_1a.solve_dual(mu, U=U)
            # this for 3 and 4
            u_r_3a = opt_rom_3a.solve(mu)
            u_r_4a = opt_rom_4a.solve(mu)
            p_r_3a = opt_rom_3a.solve_dual(mu)
            p_r_4a = opt_rom_4a.solve_dual(mu)

            J_1a = opt_rom_1a.output_functional_hat(mu, U=U, P=P)
            J_2a = opt_rom_2a.output_functional_hat(mu, U=U, P=P)
            J_3a = opt_rom_3a.output_functional_hat(mu, U=u_r_3a, P=p_r_3a)
            J_4a = opt_rom_4a.output_functional_hat(mu, U=u_r_4a, P=p_r_4a)

            J_estimator_1a = opt_rom_1a.estimate_output_functional_hat(U, P, mu)
            J_estimator_2a = opt_rom_2a.estimate_output_functional_hat(U, P, mu)
            J_estimator_3a = opt_rom_3a.estimate_output_functional_hat(u_r_3a, p_r_3a, mu)
            J_estimator_4a = opt_rom_4a.estimate_output_functional_hat(u_r_4a, p_r_4a, mu)

            J_errors_1a.append(np.abs(actual_J - J_1a))
            J_errors_2a.append(np.abs(actual_J - J_2a))
            J_errors_3a.append(np.abs(actual_J - J_3a))
            J_errors_4a.append(np.abs(actual_J - J_4a))

            rel_J_errors_1a.append(np.abs(actual_J - J_1a)/actual_J)
            rel_J_errors_2a.append(np.abs(actual_J - J_2a)/actual_J)
            rel_J_errors_3a.append(np.abs(actual_J - J_3a)/actual_J)
            rel_J_errors_4a.append(np.abs(actual_J - J_4a)/actual_J)

            J_estimators_1a.append(J_estimator_1a)
            J_estimators_2a.append(J_estimator_2a)
            J_estimators_3a.append(J_estimator_3a)
            J_estimators_4a.append(J_estimator_4a)

            effectivities_J_1a.append(J_estimator_1a/J_errors_1a[-1])
            effectivities_J_2a.append(J_estimator_2a/J_errors_2a[-1])
            effectivities_J_3a.append(J_estimator_3a/J_errors_3a[-1])
            effectivities_J_4a.append(J_estimator_4a/J_errors_4a[-1])

            # DJ
            DJ_1a = opt_rom_1a.output_functional_hat_gradient(mu, U=U, P=P)
            DJ_2a = opt_rom_2a.output_functional_hat_gradient(mu, U=U, P=P)
            DJ_3a = opt_rom_3a.output_functional_hat_gradient(mu, U=u_r_3a, P=p_r_3a)
            DJ_4a = opt_rom_4a.output_functional_hat_gradient(mu, U=u_r_4a, P=p_r_4a)

            DJ_errors_1a.append(np.linalg.norm(actual_DJ - DJ_1a))
            DJ_errors_2a.append(np.linalg.norm(actual_DJ - DJ_2a))
            DJ_errors_3a.append(np.linalg.norm(actual_DJ - DJ_3a))
            DJ_errors_4a.append(np.linalg.norm(actual_DJ - DJ_4a))

            rel_DJ_errors_1a.append(np.linalg.norm(actual_DJ - DJ_1a)/np.linalg.norm(actual_DJ))
            rel_DJ_errors_2a.append(np.linalg.norm(actual_DJ - DJ_2a)/np.linalg.norm(actual_DJ))
            rel_DJ_errors_3a.append(np.linalg.norm(actual_DJ - DJ_3a)/np.linalg.norm(actual_DJ))
            rel_DJ_errors_4a.append(np.linalg.norm(actual_DJ - DJ_4a)/np.linalg.norm(actual_DJ))

            # primal and dual
            u_r_5a = U
            p_r_5a = P
            estimate_4a = opt_rom_4a.estimate(u_r_4a, mu)
            estimate_5a = opt_rom_5a.estimate(u_r_5a, mu)

            u_r_4a_h = reductor_4a.primal.reconstruct(u_r_4a)
            u_r_5a_h = reductor_5a.primal.reconstruct(u_r_5a)

            gradient_error_4a = np.sqrt(opt_fom.opt_product.pairwise_apply2(u_h - u_r_4a_h, u_h - u_r_4a_h))
            gradient_error_5a = np.sqrt(opt_fom.opt_product.pairwise_apply2(u_h - u_r_5a_h, u_h - u_r_5a_h))

            rel_gradient_error_4a = gradient_error_4a[-1]/opt_fom.opt_product.pairwise_apply2(u_h, u_h)
            rel_gradient_error_5a = gradient_error_5a[-1]/opt_fom.opt_product.pairwise_apply2(u_h, u_h)

            eff_4a = estimate_4a[-1]/gradient_error_4a[-1]
            eff_5a = estimate_5a[-1]/gradient_error_5a[-1]

            estimate_p_4a = opt_rom_4a.estimate_dual(u_r_4a, p_r_4a, mu)
            estimate_p_5a = opt_rom_5a.estimate_dual(u_r_5a, p_r_5a, mu)

            p_r_4a_h = reductor_4a.dual.reconstruct(p_r_4a)
            p_r_5a_h = reductor_5a.dual.reconstruct(p_r_5a)

            p_gradient_error_4a = np.sqrt(opt_fom.opt_product.pairwise_apply2(p_h - p_r_4a_h, p_h - p_r_4a_h))
            p_gradient_error_5a = np.sqrt(opt_fom.opt_product.pairwise_apply2(p_h - p_r_5a_h, p_h - p_r_5a_h))

            rel_p_gradient_error_4a = p_gradient_error_4a[-1]/opt_fom.opt_product.pairwise_apply2(p_h, p_h)
            rel_p_gradient_error_5a = p_gradient_error_5a[-1]/opt_fom.opt_product.pairwise_apply2(p_h, p_h)

            eff_p_4a = estimate_p_4a[-1]/p_gradient_error_4a[-1]
            eff_p_5a = estimate_p_5a[-1]/p_gradient_error_5a[-1]

            u_errors_4a.append(gradient_error_4a)
            u_errors_5a.append(gradient_error_5a)

            rel_u_errors_4a.append(rel_gradient_error_4a)
            rel_u_errors_5a.append(rel_gradient_error_5a)

            u_estimators_4a.append(estimate_4a)
            u_estimators_5a.append(estimate_5a)

            effectivities_u_4a.append(eff_4a)
            effectivities_u_5a.append(eff_5a)

            p_errors_4a.append(p_gradient_error_4a)
            p_errors_5a.append(p_gradient_error_5a)

            rel_p_errors_4a.append(rel_p_gradient_error_4a)
            rel_p_errors_5a.append(rel_p_gradient_error_5a)

            p_estimators_4a.append(estimate_p_4a)
            p_estimators_5a.append(estimate_p_5a)

            effectivities_p_4a.append(eff_p_4a)
            effectivities_p_5a.append(eff_p_5a)

        J_errors_1as.append(J_errors_1a)
        J_errors_2as.append(J_errors_2a)
        J_errors_3as.append(J_errors_3a)
        J_errors_4as.append(J_errors_4a)

        rel_J_errors_1as.append(rel_J_errors_1a)
        rel_J_errors_2as.append(rel_J_errors_2a)
        rel_J_errors_3as.append(rel_J_errors_3a)
        rel_J_errors_4as.append(rel_J_errors_4a)

        J_estimators_1as.append(J_estimators_1a)
        J_estimators_2as.append(J_estimators_2a)
        J_estimators_3as.append(J_estimators_3a)
        J_estimators_4as.append(J_estimators_4a)

        effectivities_J_1as.append(effectivities_J_1a)
        effectivities_J_2as.append(effectivities_J_2a)
        effectivities_J_3as.append(effectivities_J_3a)
        effectivities_J_4as.append(effectivities_J_4a)

        DJ_errors_1as.append(DJ_errors_1a)
        DJ_errors_2as.append(DJ_errors_2a)
        DJ_errors_3as.append(DJ_errors_3a)
        DJ_errors_4as.append(DJ_errors_4a)

        rel_DJ_errors_1as.append(rel_DJ_errors_1a)
        rel_DJ_errors_2as.append(rel_DJ_errors_2a)
        rel_DJ_errors_3as.append(rel_DJ_errors_3a)
        rel_DJ_errors_4as.append(rel_DJ_errors_4a)

        u_errors_4as.append(u_errors_4a)
        u_errors_5as.append(u_errors_5a)

        rel_u_errors_4as.append(rel_u_errors_4a)
        rel_u_errors_5as.append(rel_u_errors_5a)

        u_estimators_4as.append(u_estimators_4a)
        u_estimators_5as.append(u_estimators_5a)

        effectivities_u_4as.append(effectivities_u_4a)
        effectivities_u_5as.append(effectivities_u_5a)

        p_errors_4as.append(p_errors_4a)
        p_errors_5as.append(p_errors_5a)

        rel_p_errors_4as.append(rel_p_errors_4a)
        rel_p_errors_5as.append(rel_p_errors_5a)

        p_estimators_4as.append(p_estimators_4a)
        p_estimators_5as.append(p_estimators_5a)

        effectivities_p_4as.append(effectivities_p_4a)
        effectivities_p_5as.append(effectivities_p_5a)

        Js.append(J)


    return J_errors_1as, rel_J_errors_1as, J_estimators_1as, effectivities_J_1as, \
           J_errors_2as, rel_J_errors_2as, J_estimators_2as, effectivities_J_2as, \
           J_errors_3as, rel_J_errors_3as, J_estimators_3as, effectivities_J_3as, \
           J_errors_4as, rel_J_errors_4as, J_estimators_4as, effectivities_J_4as, \
           DJ_errors_1as, rel_DJ_errors_1as, \
           DJ_errors_2as, rel_DJ_errors_2as, \
           DJ_errors_3as, rel_DJ_errors_3as, \
           DJ_errors_4as, rel_DJ_errors_4as, \
           Js, \
           u_errors_4as, rel_u_errors_4as, u_estimators_4as, effectivities_u_4as, \
           u_errors_5as, rel_u_errors_5as, u_estimators_5as, effectivities_u_5as, \
           p_errors_4as, rel_p_errors_4as, p_estimators_4as, effectivities_p_4as, \
           p_errors_5as, rel_p_errors_5as, p_estimators_5as, effectivities_p_5as


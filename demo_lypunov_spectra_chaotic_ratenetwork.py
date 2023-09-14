#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pylab import *
import random
import math


def rate_Lyapunov_spectra(N, g, dt, tSim, nLE, tONS, seedIC, seedNet, seedONS, dns):
    """
    This minimal demo calculates Lyapunov spectra of the dynamics of recurrent neural networks.

    Rainer Engelken 2020

    Engelken, R., Wolf, F. & Abbott, L. F. Lyapunov spectra of chaotic recurrent neural networks.
    arXiv:2006.02427 [nlin, q-bio] (2020).
    https://arxiv.org/abs/2006.02427

    Args:
        N: network size
        g: coupling strength
        dt: time discretization (in tau)
        tSim: total simulation time (in tau)
        nLE: number of Lyapunov exponents to be calculated
        tONS: orthonormalization time interval (in tau)
        seedIC: seed for random initial condition of network state
        seedNet: seed for random network realization
        seedONS: seed for random orthonormal system
        dns: flag for brute force estimate of largest Lyapunov exponent

    Returns:
        Lspectrum: Lyapunov spectrum
        LSall: normalized log of diagonal elements of R matrix
        pAll: participation ratio (localiation of first covariant Lyapunov vector)
        LambdaMaxDNS: Largest Lyapunov exponent obtained from direct numerical simulations
        LambdaMaxDNSall: All local growth rates of direct numerical simulations

    # Example use:
    rate_Lyapunov_spectra(100,4,1e-1,1e3,100,1,1,1,1,True)

    # Convergence of direct numerical vs Jacobian based max local Lyapunv exponent:
    Lspectrum,LSall,pAll,LambdaMaxDNS,LambdaMaxDNSall = rate_Lyapunov_spectra(100,4,1e-1,1e2,2,0.1,1,1,1,True)
    figure();semilogy(abs(LSall[:,0]-LambdaMaxDNSall),"k")
    ylabel(r'$|\lambda^{local}_{DNS}- \lambda^{local}_{Jacobian}|$')
    xlabel('steps')
    show()
    """

    tstart = time.time()
    assert nLE <= N, 'number of Lyapunov exponents have to be smaller than N'
    assert nLE > 0, 'number of Lyapunov exponents has to be larger than 0'
    assert dt <= 1, 'time discretization has to be smaller/equal than 1'
    assert tONS <= tSim, 'ONS interval has to be smaller than simulation time'
    assert dt <= tONS, 'dt has to be less than or equal to ONS interval'

    if tONS > 10:
        print('WARNING: ONS interval  might lead to ill-conditioned result')

    # Some default parameters:
    stepDisplay = 1000     # Interval of displaying progress
    nStepTransient = math.ceil(100/dt)  # Steps during initial transient
    tsimstart = time.time()

    # Set parameters
    nStep = math.ceil(tSim/dt)  # Total number of steps
    nstepONS = math.ceil(tONS/dt)  # Steps between orthonormalizations
    nStepTransientONS = math.ceil(nStep/10)  # Steps during transient of ONS
    nONS = math.ceil((nStepTransientONS+nStep)/nstepONS)-1  # Number of ONS steps
    h = zeros((N, nStep + nStepTransient + nStepTransientONS))  # Preallocate local fields
    np.random.seed(seedNet)  # Set seed for random network realization
    J = g*randn(N, N)/sqrt(N)  # Initialize random network realization
    J = J-diag(diag(J))  # Remove autapses
    np.random.seed(seedIC)  # Set seed for initial state
    h[:, 0] = (g-1)*randn(N)  # Initialize network state
    np.random.seed(seedONS)  # Set seed for orthonormal system
    q, r = qr(randn(N, nLE))  # Initialize orthonormal system
    diagElmR = range(0, N*nLE-1, nLE+1)  # Indices of diagonal elements or R
    LS = zeros(nLE)  # Initialize Lyapunov spectrum
    t = 0.0  # Set time to 0
    Ddiag = eye(N)*(1-dt)  # Diagonal elements of Jacobian
    LSall = zeros((nONS, nLE))  # Initialize array to store convergence of Lyapunov spectrum
    normdhAll = zeros(nONS)  # Initialize
    printCondQ = True
    pAll = zeros(nONS)
    lsidx = -1

    if dns:
        pertSz = 1e-6  # initial perturbation size for brute force max. Lyapunov exponent
        hPert = h[:, 0] + pertSz*q[:, 0]/norm(q[:, 0])
        LambdaMaxDNSall = []

    for n in range(nStep + nStepTransient + nStepTransientONS-1):
        h[:, n+1] = h[:, n]*(1-dt)+dot(J, tanh(h[:, n]))*dt  # network dynamics
        if dns:
            hPert = hPert*(1-dt)+dot(J, tanh(hPert))*dt  # perturbed network dynamics

        if (n+1 > nStepTransient):
            hsechdt = dt/cosh(h[:, n])**2  # derivative of tanh(h)*dt
            D = Ddiag + J*hsechdt  # Jacobian
            q = dot(D, q)  # evolve orthonormal system using Jacobian
            if mod(n+1, nstepONS) == 0:
                if printCondQ:  # print condition number, shouldn't be too large
                    print('log10 of cond(q): ', round(log10(cond(q)), 2))
                    printCondQ = False
                q, r = qr(q)  # performe QR-decomposition
                if nLE == 1:
                    q4 = q*q*q*q
                else:
                    q1 = q[:, 0]
                    q4 = q1*q1*q1*q1
                lsidx += 1
                pAll[lsidx] = 1.0/sum(q4)
                LSall[lsidx, :] = log(abs(diag(r)))/nstepONS/dt  # store convergence of Lyapunov spectrum
                if n + 1 > nStepTransientONS + nStepTransient:
                    LS += log(abs(diag(r)))  # collect log of diagonal elements or  R for Lyapunov spectrum
                    t += nstepONS*dt  # increment time

            if mod(n + 1, stepDisplay) == 0:  # plot progress
                if n + 1 > nStepTransient + nStepTransientONS:
                    PercentageFinished = (n + 1 - nStepTransient - nStepTransientONS)*100/nStep
                    print(round(PercentageFinished), ' % done after ', round(3.3), 's SimTime: ', round(dt*(n+1)), ' tau, std(h) =', round(std(h[:, n+1]), 2))
        if dns:  # Bruce force calculation of largest Lyapunov exponent
            dh = hPert - h[:, n+1]  # Difference of perturbed and unperturbed state
            normdh = norm(dh)  # Norm of difference
            # if normdh < dhThresholdLow or  normdh > dhThresholdUp:
            if mod(n+1, nstepONS) == 0:
                hPert = h[:, n+1] + dh*pertSz/normdh
                if n + 1 > nStepTransient:
                    normdhAll[lsidx] = normdh

    Lspectrum = LS/t  # Normalize sum of log of diagonal elements of R by total simulation time
    if dns:
        LambdaMaxDNS = sum(log(normdhAll/pertSz))/(nStep+nStepTransientONS)/dt
        LambdaMaxDNSall = log(normdhAll/pertSz)/nstepONS/dt

    subplot(221)  # Plot example trajectories
    plot(dt*arange(h.shape[1]), transpose(h[0:3, :]))
    title('example activity')
    ylabel('$x_i$')
    xlabel(r'Time ($\tau$)')
    xlim(0, dt*h.shape[1])

    subplot(223)  # Plot Lyapunov spectrum
    plot(1.0*arange(nLE)/nLE, Lspectrum, '.k')
    plot(1.0*arange(nLE)/N, zeros(nLE), ':', color=[0.5, 0.5, 0.5])
    ylabel(r'$\lambda_i (1/\tau)$')
    xlabel(r'$i/N$')
    title("Lyapunov exponents")

    subplot(222)  # Plot time-resolved first local Lyapunov exponent
    # (Jacobian-based and direct simulations)
    plot(arange(nONS)*nstepONS*dt, LSall[:, 0], 'k')
    if dns:
        timeDNS = arange(nONS)*nstepONS*dt
        plot(timeDNS, LambdaMaxDNSall, '--r')
        legend(('Jacobian-based', 'direct numerical simulations', 'Location', 'best'))
    xlabel(r'Time ($\tau$)')
    ylabel('$\lambda_i^{local}$')
    title('first local Lyapunov exponent')
    xlim(0, nONS*nstepONS*dt)

    subplot(224)  # Plot participation ratio
    plot(arange(nONS)*nstepONS*dt, pAll[:], 'k')
    xlabel(r'Time ($\tau$)')
    ylabel('P')
    title('participation ratio')
    xlim(0, nONS*nstepONS*dt)

    tend = time.time()
    print(tend-tstart, 'sec elapsed')
    tight_layout()
    show()
    return Lspectrum, LSall, pAll, LambdaMaxDNS, LambdaMaxDNSall

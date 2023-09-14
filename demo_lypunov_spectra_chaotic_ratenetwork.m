%% This minimal demo scripts calculates Lyapunov spectra of the dynamics of firing rate networks
% input parameters are:
% N: network size
% g: coupling strength
% dt: time discretization (in tau)
% tSim: total simulation time (in tau)
% nLE: number of Lyapunov exponents to be calculated
% tONS: orthonormalization time interval (in tau)
% seedIC: seed for random initial condition of network state
% seedNet: seed for random network realization
% seedONS: seed for random orthonormal system
% bs: flag for bootstrap confidence intervals of Lyapunov
% bf: flag for brute force estimate of largest Lyapunov exponent
%
% % Example use:
% close all;DemoRateLyapunovSpectra4Lee(100,4,1e-1,1e3,100,1,1,1,1,true,true);
%
% % Convergence of direct numerical vs Jacobian based max local Lyapunv exponent:
% close all;[Lspectrum,LSall,pAll,LambdaMaxBF,LambdaMaxBFall,ci] = DemoRateLyapunovSpectra02BF_LLE(100,4,1e-1,1e2,100,0.1,1,1,1,true,true);
% figure();semilogy(abs(LSall(:,1)-LambdaMaxBFall'),'k')
% ylabel('|\lambda^{local}_{BF}- \lambda^{local}_{Jacobian}|')
% xlabel('steps')
%
% % Convergence of Lyapunov exponents with time across different initial conditions:
% figure();
% plotIdx = 1:10:100
% for seedIC =1:10
% figure(1);[Lspectrum,LSall] = DemoRateLyapunovSpectra02BF_LLE(100,4,1e-1,1e3,100,1,seedIC,1,1,false,false);
% LamdaAllconv = cumsum(LSall(end/10:end,:),1)'./(1:size(LSall(end/10:end,:),1));
% figure(2);semilogx(LamdaAllconv(plotIdx,1:end)','-k');hold all;
% end
% ylabel("\lambda_i(s^{-1})"); xlabel("\tau");box off;
%title("Convergence of Lyapunov exponents across initial conditions")

function [Lspectrum,LSall,pAll,LambdaMaxBF,LambdaMaxBFall,ci] = DemoRateLyapunovSpectra4Lee(N,g,dt,tSim,nLE,tONS,seedIC,seedNet,seedONS,bs,bf)
tic
assert(nLE <= N,'number of Lyapunov exponents to calculate can''t be larger than N')
assert(nLE > 0,'number of Lyapunov exponents to calculate has to be larger than 0')
assert(dt <= 1,'time discretization can''t be larger than 1')
assert(tONS <= tSim,'orthonormalization interval has to be smaller than simulation time')
assert(bs*2*tONS <= tSim,'orthonormalization interval has to be smaller than 2 * simulation time for bootstrap')
assert(dt <= tONS,'dt has to be less than or equal to orthonormalization interval')

if tONS > 10
    disp(['WARNING: orthonormalization interval tONS = ' num2str(tONS) ' might lead to ill-conditioned result'])
end
% some default parameters:
stepDisplay = 1000;     % interval of displaying progress
nStepTransient = ceil(100/dt);  % steps during initial transient
Tstart = tic;

% set parameters
nStep = ceil(tSim/dt);        % total number of steps
nstepONS = ceil(tONS/dt);      % steps between orthonormalizations
nStepTransientONS = ceil(nStep/10); % steps during transient of ONS
nONS = ceil((nStepTransientONS+nStep)/nstepONS)-1; % number of ONS steps
h = zeros(N,nStep + nStepTransient + nStepTransientONS); % preallocate local fields
%noise = load("/home/rainer/code/python/rs/noise.csv");

rng(seedNet,'twister')  % set seed for random network realization
J = g*randn(N)/sqrt(N); % initialize random network realization
J = J-diag(diag(J));    % remove autapses
rng(seedIC,'twister')
h(:,1) = (g-1)*randn(N,1);  % initialize network state
rng(seedONS,'twister')
[Q, ~] = qr(randn(N,nLE),0);% initialize orthonormal system
diagElmR = 1:(nLE+1):nLE^2; % indices of diagonal elements or R
LS = zeros(1,nLE);      % initialize Lyapunov spectrum
t = 0;                  % set time to 0
Ddiag = eye(N)*(1-dt);  % diagonal elements of Jacobian
LSall = zeros(nONS,nLE);             % initialize array to store convergence of Lyapunov spectrum
normdhAll = zeros(1,nONS);
printCondQ = true;
pAll = zeros(nONS,1);
lsidx = 0;
if bf
    pertSz = 1e-8; %inc perturbation size for brute force max. Lyapunov exponent
end
LambdaMaxBFall = [];

for n = 2:(nStep + nStepTransient + nStepTransientONS)
    if bf &&  n-2 == nStepTransient
        hPert = h(:,n-1) + pertSz*Q(:,1)/norm(Q(:,1));
    end
    
    h(:,n) =  h(:,n-1)*(1-dt)+J*tanh(h(:,n-1))*dt; % network dynamics
    if bf &&  n-1 > nStepTransient
        hPert =  hPert*(1-dt)+J*tanh(hPert)*dt; % perturbed network dynamics
    end
    
    if n-1 > nStepTransient
        hsechdt = sech(h(:,n-1)).^2*dt; % derivative of tanh(h)*dt
        D = Ddiag + bsxfun(@times,J,hsechdt'); % Jacobian
        Q = D*Q; % evolve orthonormal system using Jacobian
        if mod(n-1,nstepONS) == 0
            if printCondQ % print condition number, shouldn't be too large
                disp(['log10 of condition number: ', num2str(round(log10(cond(Q)),2))]);
                printCondQ = false;
            end
            
            [Q, R] = qr(Q,0); % performe QR-decomposition
            D = diag(sign(diag(R)));% if you want to make sure that Q is unique
            Q = Q*D; % otherwise these 3 lines can be deleted/commented out
            %R = D*R;
            if nLE == 1
                Q4 = Q.*Q.*Q.*Q;
            else
                Q1 = Q(:,1);
                Q4 = Q1.*Q1.*Q1.*Q1;
            end
            lsidx = lsidx+1;
            pAll(lsidx) = inv(sum(Q4));
            LSall(lsidx,:) = log(abs(R(diagElmR)))/nstepONS/dt; % save convergence of Lyapunov spectrum
            if n > nStepTransientONS + nStepTransient
                LS = LS + log(abs(R(diagElmR))); % collect log of diagonal elements or  R for Lyapunov spectrum
                t = t + nstepONS*dt; % increment time
            end
        end
        if mod(n,stepDisplay) == 0 % plot progress
            if n > nStepTransient + nStepTransientONS
                PercentageFinished = (n-nStepTransient-nStepTransientONS)*100/nStep;
                disp([num2str(round(PercentageFinished)),' % done after ',num2str(round(toc(Tstart))),'s SimTime: ',num2str(round(dt*n)),' tau, std(h) = ',num2str(round(std(h(:,n)),2))])
            end
        end
        
        if bf
            dh = hPert - h(:,n);
            normdh = norm(dh);
            %if normdh < dhThresholdLow || normdh > dhThresholdUp
            if mod(n-1,nstepONS) == 0
                hPert = h(:,n) + dh*pertSz/normdh;
                if n > nStepTransient
                    normdhAll(lsidx) = normdh;
                end
            end
        end
    end
end
Lspectrum = LS/t; % normalize sum of log of diagonal elements or R by total simulation time
if bf
    LambdaMaxBF = sum(log(normdhAll/pertSz))/(nStep+nStepTransientONS)/dt;
    LambdaMaxBFall = log(normdhAll/pertSz)/nstepONS/dt;
end

figure('Position',[0,0,0.5*(1+sqrt(5))*1e3,1e3],'Color','w')
subplot(221) % plot example trajectories
plot(dt*(1:size(h,2)),h(1:5,:)')
title('example activity')
ylabel("x_i")
xlabel('Time (\tau)')
box off

subplot(223) % plot Lyapunov spectrum
plot((1:nLE)/N,Lspectrum,'.k')
hold on
plot((1:nLE)/N,zeros(nLE),':','Color',[0.5,0.5,0.5])
box off
ylabel('\lambda_i (1/\tau)')
xlabel('i/N')
title("Lyapunov exponents")

subplot(222) % plot time-resolved first local Lyapunov exponent
%(Jacobian-based and direct simulations)
plot((1:size(LSall,1))*nstepONS*dt,LSall(:,1),'k');
if bf
    hold all
    plot((1:size(LSall,1))*nstepONS*dt,LambdaMaxBFall,'--r');
    legend('Jacobian-based', 'direct simulations','Location','best')
    legend boxoff
end
xlabel('Time (\tau)')
box off
ylabel('\lambda_i^{local}')
title('first local Lyapunov exponent')
subplot(224) % plot participation ratio
plot((1:size(LSall,1))*nstepONS*dt,pAll(:,1),'k');
xlabel('Time (\tau)')
box off
ylabel('P')
title('participation ratio')

%% bootstrapped for confidence intervals of Lyapunov exponents
% note that bootstrapped confidence interval doesn't reflect systematic error
if bs
    disp('bootstrapping confidence intervals of Lyapunov spectrum')
    ci = bootci(1000,@mean,LSall(ceil(nStepTransientONS/nstepONS)+1:end,:));
    disp(['end of bootstrapping after ',num2str(round(toc)),' s'])
    subplot(223) % plot Lyapunov spectrum
    hold on
    plot((1:nLE)/N,ci','color',0.7*[1,1,1]);
end
toc


%% example use to compare to Python code:
%LambdaMaxDNSallp = load("/home/rainer/code/python/rs/LambdaMaxDNSall.csv");
%[Lspectrum,LSall,pAll,LambdaMaxBF,LambdaMaxBFall] = DemoRateLyapunovSpectra02BF_LLE_PythonComparisonNoise(100, 3, 0.1, 1e3, 100, 1, 1, 1, 1,true,true);
%semilogy(abs(LambdaMaxBFall(2:end)-LambdaMaxDNSallp'));
%%
%LSallp=load("/home/rainer/code/python/rs/LSall.csv");
%semilogy(abs(LSall(2:end,end)-LSallp(:,end)))
%semilogy(abs(LSall(2:end,1)-LSallp(:,1)))
%%

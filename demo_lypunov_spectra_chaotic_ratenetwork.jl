#### This scripts implements the full Lyapunov spectrum for chaos in rate ntetworks

if !isinteractive()
    #julia rs02LambdaMaxCluster.jl 100 10 0.1 200 100 1 1 1 1 test
    println(ARGS)
    N = parse(Int, ARGS[1])
    g = parse(Float64, ARGS[2])
    sigma = parse(Float64, ARGS[3])
    dt = parse(Float64, ARGS[4])
    tSim = parse(Float64, ARGS[5])
    nLE = parse(Int, ARGS[6])
    tONS = parse(Float64, ARGS[7])
    seedIC = parse(Int, ARGS[8])
    SeedNet = parse(Int, ARGS[9])
    seedONS = parse(Int, ARGS[10])
    subDir = ARGS[11]
end
println("loading libraries")
using PyPlot
using Bootstrap, Random, LinearAlgebra, DelimitedFiles, Statistics, Distributed, LinearAlgebra
println("done loading libraries")
#ncpus = nprocs(); LinearAlgebra.BLAS.set_num_threads(ncpus)
println("running:", @__FILE__, "\ndate: ", Libc.strftime(time()), " pid:", getpid(), " hostname:", gethostname());
flush(stdout);

function rs01(N, g, sigma, dt, tSim, nLE, tONS, seedIC, SeedNet, seedONS, subDir)
    ParaString = ("N=$N" * "g=$g" * "sigma=$sigma" * "dt=$dt" * "tSim=$tSim" * "nLE=$nLE" * "tONS=$tONS" * "seedIC=$seedIC" * "SeedNet=$SeedNet" * "seedONS=$seedONS") #saving Hash of File
    println("input parameters are:")
    println(ParaString)
    dirData = "data-rs01" * subDir * "/"
    isinteractive() ? dirData = "data-rs01" * subDir * "/" : true
    ~isdir(dirData) ? mkdir(dirData) : true
    println(dirData)
    dir = dirData
    if isfile(dir * ParaString * "tRun.dat")
        println("Output file already exists: \n", dir * ParaString * "tRun.dat")
         return dir * ParaString
    elseif isfile(dir * ParaString * "Running.dat")
        println("Output file is being simulated somewhere exists: \n", dir * ParaString * "Running.dat")
         #return dir * ParaString
    else
        println("Initializing network")
    end
    now = Libc.strftime(time())

    Fn = @__FILE__
    pid = getpid()
    host = gethostname()
    writedlm(dir * ParaString * "Running.dat", ["running:$Fn date:$now  pid:$pid" * "hostname:$host"])
    tStart = time_ns()          # start timer
    timeAfterStart() = round((time_ns() - tStart) / 1e9; digits = 1)

    #some assertions:
    nLE <= N || error("nLE has to be smaller or equal to N")
    nLE > 0 || error("number of Lyapunov exponents to calculate has to be larger than 0")
    dt <= 1 || error("time discretization can not be larger than 1")
    tONS <= tSim || error("orthonormalization interval has to be smaller than simulation time")
    #bs*2*tONS <= tSim || error("orthonormalization interval has to be smaller than 2 * simulation time for bootstrap")
    dt <= tONS || error("dt has to be less than or equal to orthonormalization interval")


    Random.seed!(SeedNet)                 # set seed of random number generator
    Jdt = randn(N, N)    # set up network topology
    rmul!(Jdt, dt * g / sqrt(N)) # with out allocating 
    diagJ = diagind(Jdt)          # diagonal indices of J    
    Jdt[diagJ] .= 0.0                # remove autapses


    nStep = ceil(Int, tSim / dt)
    nstepONS = ceil(Int,tONS / dt)
    tWarmup = max(100, tSim / 10)
    nStepTransient = ceil(Int, tWarmup / dt)
    nStepTransientONS = ceil(Int, nStep / 10) # steps during transient of ONS
    stepMeasure = 1      # frequency of measuring state
    calcλBF = true
    calcλMaxconvergence = true
    calcλLastconvergence = true
    calcλAllconvergence = true
    saveCondnumber = false

    calcDpca = true
    calcParticipationRatio = true

    plotStuff = false
    measureStates = true

    calcDpca && (measureStates = true)
    nstepONS = ceil(Int, tONS / dt)   # orthonormalization happens every nstepONS
    measureStates ? hStorage = zeros(N, ceil(Int, (nStep + nStepTransient + nStepTransientONS) / stepMeasure)) : true #save states for plotting
    h = zeros(N, nStep + nStepTransient + nStepTransientONS) # pr
    Random.seed!(seedIC)
    h[:, 1] = randn(N)              # initialize network state
    r = copy(h)
    hsech2 = zero(h)
    rPert = copy(h)
    #measureStates ? hStorage[:, 1] = h : true
    Random.seed!(seedONS)
    Qcompact = qr(randn(N, nLE)) # initialize orthonormal system
    Q = Matrix(Qcompact.Q)
    R = Matrix(Qcompact.R)
    diagR = diagind(R)          #diagonal indices of R
    LS = zeros(nLE)     # initialize Lyapunov spectrum
    stepDisplay = 10^6 ÷ N      # frequency of display
    isinteractive() ? stepDisplay = 10^6 ÷ N : true
    D = zeros(N, N)
    t = 0.0 # keep track of time to normalize Lyapunov spectrum by

    #some variables for calculating the largest Lyapunov exponent brute force:
    pertSz = 1e-8
    ΔhThresholdLow = 1e-10
    ΔhThresholdUp = 1e-4
    nCheck = 1
    Δh = zeros(N)
    Random.seed!(seedONS)
    hPert = h[:, 1] .+ pertSz * Q[:, 1] / norm(Q[:, 1])
    normΔhAll = Float64[]
    normΔhEvery = Float64[]
    n_boot = 10^4 # number of bootstraps
    cil = 0.95 # confidence interval for bootstrapping
    println("starting simulation")
    λMaxconvergence = Float64[]
    λLastconvergence = Float64[]
    λAllconvergence = Float64[]
    Condnumber = Float64[]
    prAll = Float64[]

    plotStuff && (calcλMaxconvergence == calcλBF == measureStates == true)

    function sech2!(hsech2, h::Vector)
        for i = 1:length(h)
            hsech2[i] = sech(h[i])^2
        end
        nothing
    end
    function tanh!(tanhh, h::Vector)
        for i = 1:length(h)
            tanhh[i] = tanh(h[i])
        end
        nothing
    end
    sigmadtSrt = sigma*sqrt(dt)
    noiseNow = zeros(N)

    @time for n = 2:(nStep+nStepTransient+nStepTransientONS)
        noiseNow.=sigmadtSrt*randn(N)
        h[:, n] = noiseNow .+ h[:, n-1] .* (1 - dt) + Jdt * tanh.(h[:, n-1])
        if calcλBF && n - 2 == nStepTransient
            hPert = h[:, n-1] .+ pertSz * Q[:, 1] / norm(Q[:, 1])
        end

        if calcλBF && n - 1 > nStepTransient
            hPert = noiseNow .+ hPert .* (1 - dt) + Jdt * tanh.(hPert)
        end

        if n - 1 > nStepTransient && nLE > 0
            sech2!(hsech2, h[:, n-1]::Vector)
            @inbounds for j = 1:N, i = 1:N
                D[i, j] = Jdt[i, j] * hsech2[j]
            end
            D[diagJ] .+= 1.0 .- dt
            Q = D * Q
            #if n == nStepTransient + nstepONS # display condition number after 5*nstepONS
            #    println("log10(condnumber)", log10(cond(Q)))
            #end
            if nLE > 0 && (n - 1) % nstepONS == 0
                if saveCondnumber #&& length(CondnumberLog10) < 10
                    push!(Condnumber, cond(Q))
                end
                Qcompact = qr(Q)
                Q = Matrix(Qcompact.Q)
                R = Matrix(Qcompact.R)
                if calcParticipationRatio
                    pr = 0.0
                    @inbounds for i = 1:N
                        pr += Q[i, 1] .^ 4
                    end
                    push!(prAll, inv(pr))
                end
                calcλMaxconvergence ? push!(λMaxconvergence, log.(abs.(R[1])) / tONS) : true
                calcλLastconvergence ? push!(λLastconvergence, log.(abs.(R[nLE^2])) / tONS) : true
                calcλAllconvergence ? append!(λAllconvergence, log.(abs.(R[diagR])) / tONS) : true
                if n > nStepTransient + nStepTransientONS
                    LS += log.(abs.(R[diagR]))
                    t += nstepONS * dt
                end
                if calcλBF
                    #n == nStepTransient && (Δh = hPert .- h[:, n]; normΔh = norm(Δh); hPert = h[:, n] .+ pertSz .* Δh ./ normΔh; @show normΔh) # xy check
                    Δh .= hPert .- h[:, n]
                    normΔh = norm(Δh)
                    n > nStepTransient && push!(normΔhEvery, normΔh)
                    # if normΔh < ΔhThresholdLow || normΔh > ΔhThresholdUp
                    if n % nstepONS == 0
                        hPert = h[:, n] .+ pertSz .* Δh ./ normΔh
                        n > nStepTransient && push!(normΔhAll, normΔh)
                    end
                end
            end
        end
        #measureStates && n % stepMeasure == 0 && (hStorage[:, n÷stepMeasure] = h)
        if n % stepDisplay == 0
            PercentageFinished = round((n - nStepTransient - nStepTransientONS) * 100 / nStep; digits = 1)
            println("\r", PercentageFinished, " % after ", timeAfterStart(), " s. Left:", round(Int, (nStep + nStepTransient + nStepTransientONS - n + 1) * timeAfterStart() / (n - nStepTransient - nStepTransientONS - 1)), " s. SimTime: ", round(Int, dt * n), " τ")
            flush(stdout)
        end
    end
    if nLE > 0 && nStep % nstepONS != 0
        Qcompact = qr(Q)
        Q = Matrix(Qcompact.Q)
        R = Matrix(Qcompact.R)
        LS += log.(abs.(R[diagR]))
        calcλMaxconvergence ? push!(λMaxconvergence, log.(abs.(R[1])) / tONS) : true
        calcλLastconvergence ? push!(λLastconvergence, log.(abs.(R[nLE^2])) / tONS) : true
        calcλAllconvergence ? append!(λAllconvergence, log.(abs.(R[diagR])) / tONS) : true
        normΔh = norm(Δh)
        push!(normΔhAll, normΔh)
    end

    Lspectrum = LS / t
    LambdaMaxBF = sum(log.(normΔhAll / pertSz)) / (nStep + nStepTransientONS) / dt
    LambdaMaxBFall = log.(normΔhAll ./ pertSz) ./ nstepONS ./ dt
    calcλBF && println("LambdaMaxBF  :", LambdaMaxBF)
    nLE > 0 && println("LambdaMax:", Lspectrum[1])
    calcλBF && nLE > 0 && println("LambdaMaxBF-LambdaMax:", LambdaMaxBF - Lspectrum[1])
    ###xy  calcλBF && nLE > 0 && println("LambdaMaxBF-LambdaMaxAlt2:", LambdaMaxBF - mean(λMaxconvergence[end÷10:end]))

    ####################
    ## postprocessing ##
    ####################

    if calcDpca
        println("calculating pca dimensionality I")
        covRate = cov(view(h, :, nStepTransient+1:size(h, 2))')
        eigVal = eigvals(covRate)
        eigVal = eigVal / sum(eigVal)
        eigVal = reverse(eigVal)
        relDimPCA = 100 * 1 / sum((eigVal) .^ 2) / N
        @show relDimPCA
        println("calculating pca dimensionality II")
        covRatetanh = cov(tanh.(view(h, :, nStepTransient+1:size(h, 2)))')
        eigValtanh = eigvals(covRatetanh)
        eigValtanh = eigValtanh / sum(eigValtanh)
        eigValtanh = reverse(eigValtanh)
        relDimPCAtanh = 100 * 1 / sum((eigValtanh) .^ 2) / N
        @show relDimPCAtanh
    end


    xLim = [0tSim, tSim]
    if plotStuff
        
                tPlot = (1:stepMeasure:(nStep÷stepMeasure)) * dt * stepMeasure
                subplot(121)
                plot(tPlot, h[1:10, 1:stepMeasure:(nStep÷stepMeasure)]')
                subplot(122)
                plot((1:nLE) / N, Lspectrum, ".")
                #subplot(133)
                #u,v = eig(J)
                #plot(real(u),imag(u),".")
                figure()
                subplot(211)
                plot(normΔhEvery[2:end], ".-")
                subplot(212)
                plot(λMaxconvergence, ".-")
                figure()
                tPlot = (1:stepMeasure:(nStep÷stepMeasure)) * dt * stepMeasure
                figure()
                subplot(121)
                plot(tPlot, h[1:10, 1:stepMeasure:(nStep÷stepMeasure)]')
                xlabel(L"Time ($\tau$)")
                ylabel(L"h_i")
                xlim(xLim)
                subplot(122)
                plot((1:nLE) / N, Lspectrum, ".k")
                ylabel(L"\lambda_i (1/\tau)")
                xlabel(L"i/N")
                #####title("g=$g\ SeedNet= $SeedNet\ LambdaMax: $(round(Lspectrum[1];digits=4))")

                #        tight_layout()
                figure()
                subplot(311)
                semilogy(dt * (1:length(normΔhEvery[2:end])), normΔhEvery[2:end], ".-k")
                xlabel(L"Time ($\tau$)")
                ylabel(L"\Delta h")
                xlim(xLim)
                ##### title("g=$g\ SeedNet= $SeedNet\ LambdaMax: $(round(Lspectrum[1];digits=4))")
                subplot(312)
                plot((1:length(λMaxconvergence)) * tONS, λMaxconvergence, ".-k")
                xlabel(L"Time ($\tau$)")
                ylabel(L"\lambda_1^{local} (1/\tau)")
                xlim(xLim)
                subplot(313)
                plot((1:length(prAll)) * tONS, prAll, ".-k")
                ylabel(L"P(t)")
                xlabel(L"Time ($\tau$)")
                xlim(xLim)

                figure()
                tPlot = (1:stepMeasure:(nStep÷stepMeasure)) * dt * stepMeasure
                subplot(121)
                plot(tPlot, hStorage[1:10, 1:stepMeasure:(nStep÷stepMeasure)]')
                xlabel(L"Time ($\tau$)")
                ylabel(L"h_i")
                subplot(122)
                plot((1:nLE) / N, Lspectrum, ".k")
                ylabel(L"\lambda_i (1/\tau)")
                xlabel(L"i/N")
                tight_layout()
                figure()
                subplot(311)
                plot((1:length(normΔhAll)) * tONS, log.(normΔhAll / pertSz) / tONS)
                xlabel(L"Time ($\tau$)")
                ylabel(L"\Delta h")
                subplot(312)
                plot((1:length(λMaxconvergence)) * tONS, λMaxconvergence, ".-k")
                xlabel(L"Time ($\tau$)")
                ylabel(L"\lambda_1^{local} (1/\tau)")
                subplot(313)
                plot((1:length(prAll)) * tONS, prAll, ".-k")
                ylabel(L"P(t)")
                xlabel(L"Time ($\tau$)")


        
    end
    ## bootstraped confidence interval for largest Lyapunov exponent
    nSamples = length(λMaxconvergence)
    if calcλAllconvergence && nLE > 0
        bs = bootstrap(mean, λMaxconvergence, BasicSampling(n_boot))

        bsLast = bootstrap(mean, λLastconvergence, BasicSampling(n_boot))
                BSinterval = confint(bs, PercentileConfInt(cil))
          BSintervalLast = confint(bsLast, PercentileConfInt(cil))
              println("bootstrapped confidence intervals of last Lyapunov Exponent:", BSintervalLast, " from ", nSamples, " samples")
              println("bootstrapped confidence intervals of largest Lyapunov Exponent:", BSinterval, " from ", nSamples, " samples")
    end
    ##################
    ## save results ##
    ##################
    nLE > 0 ? writedlm(dir * ParaString * "Lspectrum.dat", Lspectrum) : true
    λAllconvergence2 = reshape(λAllconvergence, nLE, div(length(λAllconvergence), nLE))

    if calcλAllconvergence && nLE > 0 && nSamples > 10

        λAllconvergenceR = reshape(λAllconvergence, nLE, div(length(λAllconvergence), nLE))

        A = [λAllconvergenceR[:, i] for i = size(λAllconvergenceR, 2)÷2:size(λAllconvergenceR, 2)]
        println("typeof(A):", typeof(A))
        bsAll = bootstrap(mean, A, BasicSampling(n_boot))
        BSinterval = confint(bsAll, BasicConfInt(cil))
        BSIntervalSpectrum = [BSinterval[i][j] for i = 1:length(BSinterval), j = 1:3]
    end

    println("nLE=", nLE)
    if nLE > 1 && calcλAllconvergence
        function getDim(x::AbstractArray)
            xMean = mean(x, dims = 2)
            Dlb = sum(cumsum(xMean, dims = 1) .> 0)
            if Dlb == length(xMean)
                return NaN
            end
            Dlb + sum(xMean[1:Dlb]) / abs.(xMean[Dlb+1])
        end

        function getH(x::AbstractArray)
            xMean = mean(x, dims = 2)
            sum(xMean[xMean.>0])
        end


        println("naive lower bound on D:", sum(cumsum(Lspectrum) .> 0))
    end

    calcλAllconvergence ? λAllconvergenceR = cumsum(λAllconvergenceR', dims = 1) ./ collect(1:size(λAllconvergenceR, 2)) : true
    calcλMaxconvergence ? writedlm(dir * ParaString * "LambdaMaxConvergence.dat", λMaxconvergence) : true
    calcλLastconvergence ? writedlm(dir * ParaString * "LambdaLastConvergence.dat", λLastconvergence) : true
    calcλAllconvergence ? writedlm(dir * ParaString * "LambdaAllConvergence.dat", λAllconvergenceR) : true
    saveCondnumber ? writedlm(dir * ParaString * "Condnumber.dat", Condnumber) : true
    calcλBF ? writedlm(dir * ParaString * "LambdaMaxBF.dat", LambdaMaxBF) : true
    calcλBF ? writedlm(dir * ParaString * "LambdaMaxBFall.dat", LambdaMaxBFall) : true
    calcDpca ? writedlm(dir * ParaString * "relDimPCAtanh.dat", relDimPCAtanh) : true
    calcDpca ? writedlm(dir * ParaString * "relDimPCA.dat", relDimPCA) : true

    rm(dir * ParaString * "Running.dat")
    tRun = round(Int, timeAfterStart())
    writedlm(dir * ParaString * "tRun.dat", tRun)
    print("-"^50, "\n End of simulation. Total simulation time:\n", round(Int, timeAfterStart()), "s\n")
    return dir * ParaString
end
if !isinteractive()
    rs01(N, g, sigma, dt, tSim, nLE, tONS, seedIC, SeedNet, seedONS, subDir)
end

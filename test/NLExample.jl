function NLExample()

    #### Required #### 

    #Initial data matric (column 1 = X values column 2 = Y values)
        cd(@__DIR__)
        #file = matopen("regression_data_1.mat");
        Initial_Data::Matrix{Float64} = load("sample_regression_data.jld2", "M");
        #close(file)
    #Function Model for what the data might fit
        Function_Model(x::Union{Vector{Float64},LinRange{Float64,Int64}},b::Vector{Float64}) = b[1] .+ b[2] .* cos.(b[3] .* x .+ b[4]);

    ##################

    #### Optional ####

        #Set to true if you want to test plots for your modle against original
        #data along with initial guess
            PlotTestMode::Bool = false;
        #Animate Option
            Animate::Bool = true;
        # Array of initial guess of b vector EG: [3; 2; 1; 1];
            Initial_Guess::Vector{Float64} = [0.1;0.1;0.1;0.1];
        # Gamma or 'step'/'speed' of regression Default: 0.001
            Gamma::Float64 = 0.001;
        # Itterations the algorithim will run Default: 10000 7e4
            Iterations::UInt64 = 2e4;

    #Run Code
    let
        M::Matrix{Float64} = Initial_Data;
        f::Function = Function_Model;
        ptm::Bool = PlotTestMode;
        b₀::Vector{Float64} = Initial_Guess;
        γ::Float64 = Gamma;
        iter::UInt64 = Iterations;
        anim::Bool = Animate;

        if ptm == true
            X::Vector{Float64} = M[:,1];
            Y::Vector{Float64} = M[:,2];
            x::LinRange{Float64,Int64} = LinRange( minimum(X) , maximum(X) ,1000);
            scatter(X, Y);
            plot(x, f(x, b₀));
            labels::Vector{String} = ["data", "Initial guess/Model"];
            xlabel('x');
            ylabel('y');
            legend(labels);
        else
            (B::Vector{Float64},MSE::Float64,RS::Float64) = makieNL(M,f,γ,iter,b₀,anim);
            println("B: "*string(B))
            println("MSE: "*string(MSE))
            println("R^2: "*string(RS));
        end
    end
    return nothing
end

NLExample()
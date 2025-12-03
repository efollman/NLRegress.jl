function gradient(fun::Function, X::Vector, h::Float64)

    g::Vector = zeros(size(X));
    
    for i=1:length(X)

        H = zeros(size(X));
        H[i] = h;
        g[i] = (fun(Vector{Float64}(X+H)) - fun(Vector{Float64}(X-H))) ./ (2*h);
        
    end

    return g
end

function gradient_descent(lossfun::Function, X₀::Vector, γ::AbstractFloat, N::UInt)
    X::Vector = X₀;
    h::Float64 = 1e-5;
    for i=1:N
        X = X - (γ .* gradient(lossfun,X,h)); #goofball shit neccesarry
    end
    return X
end

function makieNL(X::RealVector,Y::RealVector,f::Function,γ::Real,iter::Integer,b₀::RealVector,anim::Bool)

    #Reproccessing

    
    #Loss Function
    g(b::RealVector) = mean((Y .- f(X, b)) .^ 2);

    #Innitial Guess empty?
    #Running Optimization Algorithim
    hist_length::UInt64 = 1000;
    loss_hist::Observable{Vector{Float64}} = Observable{Vector{Float64}}(vec(fill(NaN,1, hist_length)));
    b_hist::Observable{Matrix{Float64}} = Observable{Matrix{Float64}}(fill(NaN,length(b₀), hist_length));
    b_gd::Observable{Vector{Float64}} = Observable{Vector{Float64}}(b₀);
    loss_gd::Observable{Float64} = Observable{Float64}(0.0);

###

    #Visualization of Result
    #tiledlayout(3, 1);
    mytheme::Attributes = Theme(
        #Axis = (
            #backgroundcolor = :gray,
        #),
    fontsize = 10,
    linewidth = 2,
    #backgroundcolor= :gray,
        
    )
    dark_latexfonts::Attributes = merge(mytheme, theme_dark(), theme_latexfonts())
    set_theme!(dark_latexfonts)
    msize::UInt64 = 5;
    lwsize::UInt64 = 2;
    roundDigits::Int64 = 3;

    F::Figure = Figure(size = (1000,600))
    x::LinRange = LinRange(minimum(X),maximum(X), 1000);
    funcY::Observable{Vector{Float64}} = @lift(f(x, $b_gd ))
    
    ax1::Axis = Axis(F[1,1])
    scatter!(X, Y; markersize = msize, label = "Initial Data")
    lines!(x, funcY;
        label = "Gradient Descent\nAlgorithim",
        linewidth = lwsize,
        color = Cycled(2)
    )
    
    # plot Loss

    #     | ||
    #    || |_

    ax2::Axis = Axis(F[2,1];
        limits = (0,iter,nothing,nothing),
    )
    x2::LinRange = LinRange(1,iter,hist_length)
    lines!(x2,loss_hist; color = Cycled(3),label = "Mean Square\nError")

    numB::Integer = length(b₀);
    
    bho1::Observable{Vector{Float64}} = @lift($b_hist[1,:])
    bho::Vector{Observable{Vector{Float64}}} = fill(bho1,size(b₀))

    for i=1:numB
        if i == 1
            ax3::Axis = Axis(F[3,1];
                limits = (0,iter,nothing,nothing)  
            )
            lines!(x2,bho[i];
            color = Cycled(3+i),
            label = "B"*string(i)
            )
        else
            bho[i] = @lift($b_hist[i,:])
            lines!(x2,bho[i];
            color = Cycled(3+i),
            label = "B"*string(i)
            )
        end
       
    end
    ax3 = ax3;
    Legend(F[1,2],ax1);
    Legend(F[2,2],ax2);
    Legend(F[3,2],ax3);
    display(F)

    ##Animate

    gdN::UInt64 = length(b₀)
    frames::UInt64 = 100
    histi::UInt64 = 1
    margain::Float64 = 0.1;
    bmax::Float64 = 0
    bmin::Float64 = 0
    brange::Float64 = 0
    losslims::Tuple{Float64,Float64} = (0.0,0.0)
    blims::Tuple{Float64,Float64} = (0.0,0.0)
    for i=1:iter
        b_gd[] = gradient_descent(g, b_gd[], γ, gdN)
        loss_gd[] = g(b_gd[]);
        if i % (iter/hist_length) == 0
            loss_hist[][histi] = loss_gd[];
            b_hist[][1:length(b_gd[]),histi] = b_gd[];
            histi = histi + 1
        end
        if (anim == true) && ((i % (iter ÷ frames)) == 0)
            notify(b_gd)
            notify(b_hist)
            notify(loss_gd)
            notify(loss_hist)
            
            if loss_gd[] < losslims[1] + (losslims[2]-losslims[1])*margain
                losslims = (loss_gd[] - (losslims[2]-losslims[1])*margain, losslims[2])
                ylims!(ax2,losslims)
            end

            if loss_gd[] > losslims[2] - (losslims[2]-losslims[1])*margain
                losslims = (losslims[1], loss_gd[] + (losslims[2]-losslims[1])*margain)
                ylims!(ax2,losslims)
            end

            bmax = maximum(b_gd[])
            bmin = minimum(b_gd[])
            brange = bmax-bmin;
        
            if bmin < blims[1] + brange*margain
                blims = (bmin - brange*margain, blims[2])
                ylims!(ax3,blims)
            end

            if bmax > blims[2] - brange*margain
                blims = (blims[1], bmax + brange*margain)
                ylims!(ax3,blims)
            end
        
            yield()
        end
    end

    B::Vector{Float64} = b_gd[]
    MSE::Float64 = g(b_gd[])
    RS::Float64 = 1 - MSE / mean((Y .- mean(Y)) .^ 2);
    
    B = round.(B, digits=roundDigits)
    MSE = round(MSE, digits=roundDigits)
    RS = round(RS, digits=roundDigits)

    return B, MSE, RS

end
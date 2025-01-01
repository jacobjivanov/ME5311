# burning_ship.jl

import Pkg; 
Pkg.add("PyPlot"); Pkg.add("LaTeXStrings");

function N_func(c)
    z = 0 + 0*im; n = 0

    while n < 200
        z = ((abs(real(z))) 
        + (abs(imag(z))*im))^2 - c;
        
        if abs(z) >= 200 
            break 
        end
        n += 1
    end
    return n
end

Mx = 1000; My = 1000;

# figure 1, 2 region
x_min = -2; x_max = +2;
y_min = -2; y_max = +2;

# figure 3 region
# x_min = -1.25; x_max = -0.75;
# y_min = +1.25; y_max = +1.75;

x = LinRange(x_min, x_max, Mx);
y = LinRange(y_min, y_max, My);

N = Matrix{Float64}(undef, Mx, My);
binN = Matrix{Float64}(undef, Mx, My);
logN = Matrix{Float64}(undef, Mx, My);

for i = 1:Mx
    for j = 1:My
        N[i, j] = N_func(x[i] + im*y[j]);
        
        if N[i, j] >= 150
            binN[i, j] = 1
        else
            binN[i, j] = 0
        end

        logN[i, j] = log(N[i, j]);
    end
end

using PyPlot
using LaTeXStrings

# Julia is column-major, whereas Python Numpy is row-major. As a result, the array we pass to plt.pcolor needs to be transposed
pygui(true); # hides/shows the plot
fig, ax = subplots(1, 1)

# figure 1
binN_plot = ax.pcolor(x, y, transpose(binN), cmap = "Greys");

# figure 2, 3
# logN_plot = ax.pcolor(x, y, transpose(logN), cmap = "inferno");
# fig.colorbar(logN_plot);

ax.set_aspect("equal")
ax.set_xlabel(L"$x$");
ax.set_ylabel(L"$y$");
ax.set_title("Burning Ship Set, 200 Iterations");

savefig("HW1/burning_ship_set.png", dpi = 200)
# savefig("HW1/burning_ship.png", dpi = 200)
# savefig("HW1/burning_ship_zoom.png", dpi = 200)
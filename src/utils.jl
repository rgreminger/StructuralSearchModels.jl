# Generate products 
function generate_products(n_sessions; 
    n_products = 1_000_000,
    n_products_per_session = 30,
    distribution = Normal(), 
    outside_option = true,
    kwargs...) 

    # Set seed 
    set_seed(kwargs)

    # Product ids, 0 will be outside option 
    pid = collect(1:n_products) 

    # Draw product characteristics for each consumer 
    product_characteristics = rand(distribution, n_products, 1) 

    # Draw products per session randomly from set of products for each session 
    product_ids =   if outside_option
						[ vcat(0 , rand(pid, n_products_per_session)) for i in 1:n_sessions]
					else
						[ rand(pid, n_products_per_session) for i in 1:n_sessions]
					end

    # Gather characteristics for each product
    product_characteristics =   if outside_option
									[ hcat(vcat(zeros(1, size(product_characteristics, 2)), product_characteristics[pids[2:end], :]), vcat(1.0, zeros(length(pids)-1))) for pids in product_ids]
								else
									[ product_characteristics[pids, :] for pids in product_ids]
								end

    return product_ids, product_characteristics
end 


# Get chunk size
function get_chunks(n)
    chunk_size = max(1, n ÷ (TASKS_PER_THREAD * Threads.nthreads()))
    data_chunks = Iterators.partition(1:n, chunk_size) 
    return chunk_size, data_chunks
end

# Functional form definitions 
function get_functional_form(s::String)
	if s == "" 
		(Ξ, ρ, pos) -> Ξ 
	elseif s == "linear"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * pos 
	elseif s == "log"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * _LP[pos + 1]
	elseif s == "exp"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * _EP[pos + 1]
	else
		error("zdfun not correctly specified")
	end
end

# Set seed 
function set_seed(kwargs)
    # Set seed if given 
	seed = get(kwargs, :seed, nothing) # if no seed set, just draw one
	if !isnothing(seed)
		Random.seed!(seed)
	end
    return nothing 
end

# Take or generate draw
@inline function take_or_generate_draw!(draws, distribution::Distribution, i, j, store_draw)
	if isnothing(draws) 
		return rand(distribution)::Float64
	elseif store_draw 
		new_draw = rand(distribution)::Float64
		draws[i, j] = new_draw
		return new_draw
	else
		return draws[i, j] 
	end
end

function get_precomputed_draws(kwargs) 
	draws_u0 = get(kwargs, :draws_u0, nothing)
	draws_e = get(kwargs, :draws_e, nothing)
	draws_v = get(kwargs, :draws_v, nothing)
	draws_w = get(kwargs, :draws_w, nothing)

	return (draws_u0, draws_e, draws_v, draws_w)
end

function get_precomputed_draws_indexed(kwargs, s) 

	draws_u0, draws_e, draws_v, draws_w = get_precomputed_draws(kwargs)

	draws_u0 = isnothing(draws_u0) ? nothing : @views draws_u0[s]
	draws_e = isnothing(draws_e) ? nothing : @views draws_e[s]
	draws_v = isnothing(draws_v) ? nothing : @views draws_v[s]
	draws_w = isnothing(draws_w) ? nothing : @views draws_w[s]

	return (draws_u0, draws_e, draws_v, draws_w)
end

@inline function replace_typemin_draws!(draws, dist::Distribution)
	@inbounds for i in eachindex(draws)
		if isequal(draws[i], typemin(eltype(draws)))
			draws[i] = rand(dist)
		end
	end
end

	

##################################################################################
# Computations for bivariate normal 

# Source: https://github.com/JuliaStats/StatsFuns.jl/blob/60dc61c63ba74d0da0e1000a4b3b785a6badacf5/src/tvpack.jl
# (open pull request on StatsFuns.jl)
# Removed unnecessary parts and adapted to accept Reals (necessary for AD)

# Written in Julia by Andreas Noack Jensen
# January 2013
#
# Translation of Fortran file tvpack.f authored by
#
# Alan Genz
# Department of Mathematics
# Washington State University
# Pullman, WA 99164-3113
# Email : alangenz@wsu.edu
#
# Original source available from
# http://www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f



# This function is based on the method described by
#     Drezner, Z and G.O. Wesolowsky, (1989),
#     On the computation of the bivariate normal integral,
#     Journal of Statist. Comput. Simul. 35, pp. 101-107,
# 		with major modifications for double precision, and for |R| close to 1.


@inline normcdf(x) = Distributions.normcdf(x)

const bvncdf_w_array = [0.1713244923791705e+00 0.4717533638651177e-01 0.1761400713915212e-01;
     					0.3607615730481384e+00 0.1069393259953183e+00 0.4060142980038694e-01;
     					0.4679139345726904e+00 0.1600783285433464e+00 0.6267204833410906e-01;
     					0.0 				   0.2031674267230659e+00 0.8327674157670475e-01;
     					0.0					   0.2334925365383547e+00 0.1019301198172404e+00;
     					0.0					   0.2491470458134029e+00 0.1181945319615184e+00;
     					0.0					   0.0					  0.1316886384491766e+00;
     					0.0					   0.0					  0.1420961093183821e+00;
     					0.0					   0.0					  0.1491729864726037e+00;
     					0.0					   0.0					  0.1527533871307259e+00]

const bvncdf_x_array = [-0.9324695142031522e+00 -0.9815606342467191e+00 -0.9931285991850949e+00;
						-0.6612093864662647e+00 -0.9041172563704750e+00 -0.9639719272779138e+00;
						-0.2386191860831970e+00 -0.7699026741943050e+00 -0.9122344282513259e+00;
						 0.0 				    -0.5873179542866171e+00 -0.8391169718222188e+00;
						 0.0 				    -0.3678314989981802e+00 -0.7463319064601508e+00;
						 0.0 				    -0.1252334085114692e+00 -0.6360536807265150e+00;
						 0.0 				    0.0 				    -0.5108670019508271e+00;
						 0.0 				    0.0 				    -0.3737060887154196e+00;
						 0.0 				    0.0 				    -0.2277858511416451e+00;
						 0.0 				    0.0 				    -0.7652652113349733e-01]

function bvnuppercdf(dh::T, dk::T, r::Tf)::T where {T<:Real,Tf}

	if r < -one(T) || r > one(T)
		throw(DomainError("r must be ∈ [-1,1]"))
	end

	if isfinite(dh) && isfinite(dk)
		if abs(r) < 0.3
		   ng = 1
		   lg = 3
		elseif abs(r) < 0.75
		   ng = 2
		   lg = 6
		else
		   ng = 3
		   lg = 10
		end
		h = dh
		k = dk
		hk = h*k
		bvn = 0.0
		if abs(r) < 0.925
		   	if abs(r) > 0
		      	hs = (h * h + k * k) * 0.5
		      	asr = asin(r)
		      	for i = 1:lg
		         	for j = -1:2:1
		            	sn = sin(asr * (j * bvncdf_x_array[i, ng] + 1.0) * 0.5)
		            	bvn += bvncdf_w_array[i, ng] * exp((sn * hk - hs) / (1.0 - sn*sn))
		        	end
		      	end
		      	bvn *= asr / (4.0pi)
		   	end
		   	bvn += normcdf(-h) * normcdf(-k)
		else
		   	if r < 0
		      	k = -k
		      	hk = -hk
		   	end
		   	if abs(r) < 1
		      	as = (1.0 - r) * (1.0 + r)
		      	a = sqrt(as)
		      	bs = (h - k)^2
		      	c = (4.0 - hk) * 0.125
		      	d = (12.0 - hk) * 0.0625
		      	asr = -(bs / as + hk) * 0.5
		      	if ( asr > -100 )
		      		bvn = a * exp(asr) * (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as * as / 5.0)
		      	end
		      	if -hk < 100
		         	b = sqrt(bs)
		         	bvn -= exp(-hk * 0.5) * sqrt(2.0pi) * normcdf(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
		      	end
		     	a /= 2.0
			    for i = 1:lg
		         	for j = -1:2:1
		            	xs = (a * (j*bvncdf_x_array[i, ng] + 1.0))^2
		            	rs = sqrt(1.0 - xs)
		            	asr = -(bs / xs + hk) * 0.5
		            	if asr > -100
		               		bvn += a * bvncdf_w_array[i, ng] * exp(asr) * (exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs * (1.0 + d * xs)))
		            	end
		         	end
		        end
		      	bvn /= -2.0pi
		   	end
		   	if r > 0
		      	bvn += normcdf(-max(h, k))
		   	else
		      	bvn = -bvn
		      	if k > h
		      		bvn += normcdf(k) - normcdf(h)
		      	end
			end
		end
		return bvn

	# here we either have 0 or the marginal
	else
		if dh == typemax(T) || dk == typemax(T)
			return zero(T)
		elseif dh == typemin(T)
			return normcdf(-dk)
		elseif dk == typemin(T)
			return normcdf(-dh)
		else
			return zero(T)
		end
	end
end

@inline bvncdf(dh, dk, r) = bvnuppercdf(-dh, -dk, r)


##################################################################################
# Truncation of random variables -> use own functions for normal to avoid rejection sampling 


# Normal distribution: helper functions 
@inline cdf_n(z::T) where T <: Real = Distributions.erfc(-z * invsqrt2)/2

@inline function invcdf_n(z::T) where T <: Real
	# Note: For z very close to 1 or zero, will return Inf
	# Fixing bounds helps calculate gradient (and introduces little inaccuracy)
	if z >= 1-eps(T)
		return sqrt2*erfinv(1-eps(T))
	elseif z <= eps(T) 
		return sqrt2*erfinv(2*eps(T)-1)
	else
		return sqrt2*erfinv(2*z-1)
	end
end

"""
	rand_trunc(d::Normal, lb::T, ub::T) where T

Take a random draw from a truncated normal distribution. 
"""
@inline function rand_trunc(d::Normal,lb::T,ub::T) where T

	@assert d.μ == 0 
	if lb >= ub 
		return lb
	end
	# Differ whether in upper or lower tail. By symmetry same, but works more accurately. Further tests an improvements welcome. 
    if lb > 0 && ub > 0
        l = cdf(d, -ub)
        u = cdf(d, -lb)
        if l == u > 0 
            u += eps(u)
        end
		r = l + rand() * (u - l)
		q = quantile(d, max(min(r, 1-1e-16), 1e-300)) # imposing bounds to avoid numerical issues in gradient leading to nan when inf from quantile
        return max(min( - q, ub), lb)
    else 
        l = cdf(d, lb)
        u = cdf(d, ub)
        if l == u > 0 
            u += eps(u)
        end

		r = l + rand() * (u - l)
		q = quantile(d, max(min(r, 1-1e-16), 1e-300)) # imposing bounds to avoid numerical issues in gradient leading to nan when inf from quantile
        return max(min(q, ub), lb)
    end
end

# @inline function rand_trunc(d::Gumbel,lb::T,ub::T) where T
# 	if lb >= ub 
# 		return lb 
# 	end
# 	if ub < -1e6 || lb > 1e6  || abs(ub-lb) < 1e-8
# 		return (ub-lb) / 2 
# 	end

# 	tp = trunc_cdf(d,lb,ub)
# 	r = if tp > sqrt(eps(T)) 
# 			c1 = lb < -1e6 ? zero(T) : cdf(d,lb) 
# 			rtcdf = rand(T)*trunc_cdf(d,lb,ub)
# 			t = isnan(c1) ? rtcdf : rtcdf + c1
# 			# note: cdf(d,lb) gives nan if lb too small because of gumbel cdf
# 			# so accounting for c1 helps AD 
# 			d.μ - d.θ*log(-log(t))
# 		else 
# 			loglcdf = max(logcdf(d,lb),-1e100)
# 			logtp = log(tp) 
# 			invlogcdf(d, Distributions.logaddexp(loglcdf, logtp - randexp(T)))
# 		end
# 	r = max(min(r,ub),lb)
# 	return r::T	
# end


"""
	trunc_cdf(d::Uniform, lb::T, ub::T) where T

Calculate the cumulative distribution function of a truncated uniform distribution.
"""
@inline function trunc_cdf(d::Uniform, lb::T, ub::T) where T <: Real
	if ub <= minimum(d) || lb >= maximum(d)
		return zero(T)
	end
	return cdf(d, ub) - cdf(d, lb)
end

@inline function rand_trunc(d::Uniform,lb::T,ub::T) where T <: Real
	if ub <= minimum(d) || lb >= maximum(d)
		return zero(T)
	end
	if lb >= ub || isnan(ub)
		return lb 
	elseif isnan(lb) 
		return ub 
	end
	return rand(truncated(d,lb,ub))
end


"""
	function rand_trunc(d::Distribution, lb::T, ub::T) where T

Calculate the cumulative distribution function of a generic truncated distribution.
"""
@inline function rand_trunc(d::Distribution, lb::T, ub::T) where T
	if ub <= minimum(d) || lb >= maximum(d)
		return zero(T)
	end
	if lb >= ub 
		return lb 
	end

	tp = trunc_cdf(d,lb,ub)
	loglcdf = max(logcdf(d,lb),-1e100)

	r = if tp > sqrt(eps(T) )
			c1 = lb < -1e12 ? zero(T) : cdf(d,lb) 
			rtcdf = rand(T)*tp
			t = isnan(c1) ? rtcdf : rtcdf + c1
			# note: cdf(d,lb) gives nan if lb too small because of gumbel cdf
			# so accounting for c1 helps AD 
			quantile(d, min(exp(loglcdf) + t,1))
		else 
			invlogcdf(d, Distributions.logaddexp(loglcdf, log(tp) - randexp()))
		end
	r = max(min(r,ub),lb)
	return r::T	
end

@inline function trunc_cdf(d, lb::T, ub::T) where T <: Real 
	# Do calculation in log-scale to avoid numerical issues
	# Note: follows truncate.jl in Distributions package 
	# introduces additional max(..) to help AD avoid NaN
	if ub <= minimum(d) || lb >= maximum(d) || lb >= ub 
		return zero(T)
	end
	
	x = max(- 1e100, logcdf(d,lb))
	y = max(- 1e100, logcdf(d,ub))

	tp = exp(Distributions.logsubexp(x,y))
	return max(1e-300, tp)
end 


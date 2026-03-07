module TrainRun

using ArgParse
using CSV
using DataFrames
using CairoMakie
using GLMakie
using YAML

export Train, Path, TrainRunSimulation
export plot_multi_profile

const G = 9.81  # Acceleration due to gravity (m/s^2)
const KMH_TO_MS = 1 / 3.6
const MS_TO_KMH = 3.6

Base.@kwdef struct Train
  name::String
  mass::Float64
  adh_mass::Float64
  davis_a::Float64
  davis_b::Float64
  davis_c::Float64
  rotational_inertia::Float64
  braking_curve_table::Vector{Tuple{Float64, Float64}}
  tractive_effort_table::Vector{Tuple{Float64, Float64}}
end

Base.@kwdef struct Path
  positions::Vector{Float64}
  speed_limits::Vector{Float64}
  slopes::Vector{Float64}
  curves::Vector{Float64}
  station_masks::Vector{Bool}
  station_pos = Vector{Float64}
  station_names = Vector{String}
end

Base.@kwdef struct TrainRunSimulation
  train::Train
  path::Path
  profile::DataFrame
end

function parse_commandline()
  s = ArgParseSettings(
    description = "TrainRun.jl - Train Run Simulation Engine",
    version = "1.0.0",
    add_version = true
  )

  @add_arg_table! s begin
    "--geometry", "-g"
    help = "Input CSV track geometry file containing pk_m, vel_kmh, curve_m, slope_‰, station_name"
    arg_type = String
    required = true

    "--train", "-t"
    help = "Input YAML file containing rolling stock information"
    arg_type = String
    required = true

    "--reverse", "-r"
    help = "Process the track direction in reverse"
    action = :store_true

    "--ppkm"
    help = "Number of discretization points per kilometer"
    arg_type = Int
    default = 1000

    "--dwell"
    help = "Dwell time at stations in seconds"
    arg_type = Float64
    default = 0.0

    "--out", "-o"
    help = "Outputs the plot to a PDF/PNG/SVG file"
    arg_type = String

    "--show", "-s"
    help = "Show plots"
    action = :store_true
  end

  return parse_args(s)
end

"""
Constructor to create a Train directly from a YAML file.
"""
function Train(filepath::String)
  data = YAML.load_file(filepath)

  braking_curve_table = Tuple{Float64, Float64}[]
  for (v, a) in data["braking_curve_table"]
    push!(braking_curve_table, (v*KMH_TO_MS, a))
  end

  tractive_effort_table = Tuple{Float64, Float64}[]
  for (v, f) in data["tractive_effort_table"]
    push!(tractive_effort_table, (v*KMH_TO_MS, f))
  end

  return Train(
    name = data["name"],
    mass = data["mass"],
    adh_mass = data["adh_mass"],
    davis_a = data["davis_a"],
    davis_b = data["davis_b"],
    davis_c = data["davis_c"],
    rotational_inertia = data["rotational_inertia"],
    braking_curve_table = braking_curve_table,
    tractive_effort_table= tractive_effort_table
  )
end

"""
Constructor to create a Path directly from a CSV file.
"""
function Path(filepath::String; ppkm::Int=100, is_reversed::Bool=false)
  df = CSV.read(filepath, DataFrame)

  # Column processing
  if ncol(df) != 5
    error("CSV must contain 5 columns: pk_m, vel_kmh, curve_m, slope_‰, station_name")
  end
  rename!(df, 1 => :x_start, 2 => :v_limit, 3 => :curve, 4 => :slope, 5 => :station)
  df.v_limit .= df.v_limit .* KMH_TO_MS

  # File Validation
  for i in 1:nrow(df)-1
    if df.x_start[i] >= df.x_start[i+1]
      error("Rows $(i+1),$(i+2): segment end must be strictly greater than segment start.")
    end

    if df.v_limit[i] < 0
      error("Row $(i+1): speed limit cannot be negative.")
    end
  end

  # High-Resolution Discretization
  pos_segs = Vector{Float64}[]
  vlim_segs = Vector{Float64}[]
  curve_segs = Vector{Float64}[]
  slope_segs = Vector{Float64}[]
  is_station_segs = Vector{Bool}[]

  # Keep track of stations
  station_pos = Float64[]
  station_names = String[]

  for i in 1:nrow(df)-1
    xi = Float64(df.x_start[i])
    xf = Float64(df.x_start[i+1])
    vlim = Float64(df.v_limit[i])
    curve = Float64(df.curve[i])
    slope = Float64(df.slope[i])
    station = df.station[i]

    # Handle station stops
    is_station = !ismissing(station) && strip(string(station)) != "nan" && strip(string(station)) != ""
    if is_station
      push!(station_pos, xi)
      push!(station_names, station)
    end

    # Generate spatial points
    dx = xf - xi
    ppm = ppkm / 1000.0
    n_points = ceil(Int, dx * ppm)
    pos = collect(range(xi, stop=xf, length=n_points+1))[1:end-1]

    push!(pos_segs, pos)
    push!(vlim_segs, fill(vlim, n_points))
    push!(curve_segs, fill(curve, n_points))
    push!(slope_segs, fill(slope, n_points))
    push!(is_station_segs, fill(is_station, n_points))
  end

  # Flatten the arrays
  positions = reduce(vcat, pos_segs)
  speed_limits = reduce(vcat, vlim_segs)
  curves = reduce(vcat, curve_segs)
  slopes = reduce(vcat, slope_segs)
  station_masks = reduce(vcat, is_station_segs)

  push!(positions, last(df.x_start))
  push!(speed_limits, last(df.v_limit))
  push!(curves, last(df.curve))
  push!(slopes, last(df.slope))
  push!(station_masks, true)
  push!(station_pos, last(df.x_start))
  push!(station_names, last(df.station))

  # Revert track geometry
  if is_reversed
    pos_offset = positions[end] + positions[1]
    positions = pos_offset .- reverse(positions)
    reverse!(speed_limits)
    reverse!(slopes)
    reverse!(curves)
    reverse!(station_masks)
    slopes .*= -1.0
    curves .*= -1.0

    station_pos = pos_offset .- reverse(station_pos) 
    reverse!(station_names)
  end

  return Path(positions, speed_limits, slopes, curves, station_masks, station_pos, station_names)
end

"""
Constructor to run the simulation.
"""
function TrainRunSimulation(train::Train, path::Path)
  v_fwd = forward_pass(train, path)
  v_bwd = backward_pass(train, path)
  v_final = min.(v_fwd, v_bwd)

  n = length(path.positions)
  time = zeros(Float64, n)
  energy = zeros(Float64, n)
  forces = zeros(Float64, n)

  # Integrate Time and Energy over the final profile
  @inbounds for i in 1:(n - 1)
    v1, v2 = v_final[i], v_final[i+1]
    v_avg = (v1 + v2) / 2.0
    ds = path.positions[i+1] - path.positions[i]

    dt = v_avg > 0 ? ds / v_avg : 0.0
    time[i+1] = time[i] + dt

    acceleration = ds > 0 ? (v2^2 - v1^2) / (2 * ds) : 0.0
    R_roll = rolling_resistance(train, v_avg)
    R_slope = slope_resistance(path, train.mass, i)
    R_curve = curve_resistance(path, train.mass, i)

    force = effective_mass(train) * acceleration + R_roll + R_slope + R_curve

    if force >= 0
      forces[i] = min(force, max_tractive_effort(train, v_avg))
      energy[i+1] = energy[i] + (forces[i] * ds)
    else
      # TODO: what happens if the acceleration calculated before is
      # required to brake safely. In that case the max force calculated
      # next will not be enough. Am I making any sense?
      forces[i] = max(force, -max_braking_effort(train, v_avg))
      energy[i+1] = energy[i]
    end
  end

  profile_df = DataFrame(
    distance_km = path.positions .* 0.001,
    velocity_kmh = v_final .* MS_TO_KMH,
    time_min = time ./ 60,
    force_kN = forces ./ 1e3,
    energy_MJ = energy ./ 1e6
  )

  return TrainRunSimulation(train, path, profile_df)
end


"""
Accounts for the rotational inertia.

See Opentrack Manual section 5.3.2 B - Acceleration Resistance
"""
effective_mass(t::Train) = t.mass * (1 + 0.01 * t.rotational_inertia)

"""
Davis equation for baseline train resistance.
R = a + bv + cv^2 = a + v(b + cv)
"""
rolling_resistance(t::Train, v::Float64) = t.mass * G * (t.davis_a + v*MS_TO_KMH * (t.davis_b + t.davis_c * v*MS_TO_KMH)) * 0.001

"""
Resistance due to vertical gradients
R = mgsin(θ) = mgtan(θ) = mg I/1000; where I is the slope in "per mille".

See Opentrack Manual section A-2: Distance Resistance: Gradient Resistance
"""
slope_resistance(p::Path, mass::Float64, index::Int) = mass * G * p.slopes[index] * 0.001

"""
Resistance due to horizontal gradients

See Opentrack Manual section A-2: Distance Resistance: Curve Resistance
See also: https://open-rails.readthedocs.io/en/latest/physics.html#curve-resistance-theory
"""
function curve_resistance(p::Path, mass::Float64, index::Int)
  curv_res = 0.0

  curv = abs(p.curves[index])
  if curv > 0
    if curv >= 300
      denom = max(1.0, curv - 55)
      specific_resistance = 650 / denom
    else
      denom = max(1.0, curv - 30)
      specific_resistance = 500 / denom
    end
    curv_res = mass * G * specific_resistance * 0.001
  end

  return curv_res
end

"""
Retrieve deceleration from braking curve map.

See https://www.era.europa.eu/domains/european-rail-traffic-management-system/braking-curves
"""
function max_braking_effort(t::Train, v::Float64)
  table = t.braking_curve_table

  if v >= table[end][1]       # Min deceleration
    deceleration = table[end][2]
  else
    idx = findfirst(row -> v < row[1], table)
    if idx !== nothing
      deceleration = table[idx][2]
    else
      # This should never happen if the table is valid
      deceleration = table[1][1]
    end
  end
  braking_force = t.mass * deceleration

  # Adhesion limit (Curtius-Kniffler equation)
  friction_coeff = 0.161 + 2.1 / (v + 12.2)
  weather_coeff = 1.25 # Normal track conditions: 125%
  adhesion_limit = t.adh_mass * G * friction_coeff * weather_coeff

  return min(braking_force, adhesion_limit)
end

"""
Interpolate tractive effort from the velocity-force map.

For adhesion limits the empirical Curtius-Kniffler equation is used.

See Opentrack Manual section 5.2.4 Adhesion Behaviour
See also: https://open-rails.readthedocs.io/en/latest/physics.html#adhesion-of-locomotives-settings-within-the-wagon-section-of-eng-files
"""
function max_tractive_effort(t::Train, v::Float64)
  # TODO: check table validity: sort by speed and validate regimes are well defined
  table = t.tractive_effort_table

  if v >= table[end][1]       # Max allowed speed
    traction_force = 0.0
  elseif v <= table[1][1]     # Constant force regime
    traction_force = table[1][2]
  else                        # Constant power regime
    idx = findfirst(row -> v < row[1], table)
    if idx !== nothing
      v1, f1 = table[idx-1]
      v2, f2 = table[idx]
      p1 = f1 * v1
      p2 = f2 * v2
      # Make a linear interpolation of power in case it
      # is not exactly constant in the current segment:
      p = p1 + (p2 - p1) / (v2 - v1) * (v - v1)

      traction_force = p / v
    else
      # This should never happen if the table is valid
      traction_force = 0.0
    end
  end

  # Adhesion limit (Curtius-Kniffler equation)
  friction_coeff = 0.161 + 2.1 / (v + 12.2)
  weather_coeff = 1.25 # Normal track conditions: 125%
  adhesion_limit = t.adh_mass * G * friction_coeff * weather_coeff

  return min(traction_force, adhesion_limit)
end


function forward_pass(train::Train, path::Path)
  n = length(path.positions)
  station_masks = path.station_masks
  speed_limits = path.speed_limits
  v_train_lim = train.tractive_effort_table[end][1]
  v_fwd = zeros(Float64, n)

  @inbounds for i in 1:(n-1)
    v_curr = v_fwd[i]

    F_t = max_tractive_effort(train, v_curr)
    R_roll = rolling_resistance(train, v_curr)
    R_slope = slope_resistance(path, train.mass, i)
    R_curve = curve_resistance(path, train.mass, i)

    # F = ma
    F_net = F_t - R_roll - R_slope - R_curve
    a = F_net / effective_mass(train)

    # Kinematic update
    ds = path.positions[i+1] - path.positions[i]
    v_next_sq = v_curr^2 + 2 * a * ds

    v_next = v_next_sq > 0 ? sqrt(v_next_sq) : 0.0
    v_track_lim = station_masks[i+1] ? 0.0 : speed_limits[i+1]
    v_fwd[i+1] = min(v_next, v_track_lim, v_train_lim)
  end

  return v_fwd
end

function backward_pass(train::Train, path::Path)
  n = length(path.positions)
  station_masks = path.station_masks
  speed_limits = path.speed_limits
  v_train_lim = train.tractive_effort_table[end][1]
  v_bwd = zeros(Float64, n)

  @inbounds for i in n:-1:2
    v_curr = v_bwd[i]

    F_b = max_braking_effort(train, v_curr)
    R_roll = rolling_resistance(train, v_curr)
    R_slope = slope_resistance(path, train.mass, i)
    R_curve = curve_resistance(path, train.mass, i)

    F_net = F_b + R_roll - R_slope + R_curve
    a = F_net / effective_mass(train)

    ds = path.positions[i] - path.positions[i-1]
    v_prev_sq = v_curr^2 + 2 * a * ds

    v_prev = v_prev_sq > 0 ? sqrt(v_prev_sq) : 0.0
    v_track_lim = station_masks[i-1] ? 0.0 : speed_limits[i-1]
    v_bwd[i-1] = min(v_prev, v_track_lim, v_train_lim)
  end

  return v_bwd
end

function plot_multi_profile(
  profile::DataFrame,
  path::Path,
  train::Train;
  out_filepath::Union{String, Nothing}=nothing,
  show::Bool=true,
)

  if !show && out_filepath === nothing
    return nothing
  end

  dist_km = profile.distance_km
  speeds_kmh = profile.velocity_kmh
  energy_MJ = profile.energy_MJ
  times_min = profile.time_min
  slopes = path.slopes
  curves = path.curves
  station_pos_km = path.station_pos .* 0.001
  station_names = path.station_names
  speed_limits_kmh = path.speed_limits .* MS_TO_KMH

  # Initialize Figure
  fig = Figure(size = (1200, 800))

  # Vertically stacked axes
  ax1 = Axis(fig[1, 1], ylabel="Slope [‰]")
  ax2 = Axis(fig[2, 1], ylabel="Curv. [1/m]")
  ax3 = Axis(fig[3, 1], ylabel="Vel. [km/h]")
  ax4 = Axis(fig[4, 1], ylabel="Energy [MJ]")
  ax5 = Axis(fig[5, 1], ylabel="Time [min]", xlabel="Distance [km]")
  # Dummy top axes for station names
  ax1_top = Axis(fig[1, 1], xaxisposition=:top)
  ax2_top = Axis(fig[2, 1], xaxisposition=:top)
  ax3_top = Axis(fig[3, 1], xaxisposition=:top)
  ax4_top = Axis(fig[4, 1], xaxisposition=:top)
  ax5_top = Axis(fig[5, 1], xaxisposition=:top)

  # Inclination profile
  # Fill between 0 and positive (red), 0 and negative (green)
  band!(ax1, dist_km, 0.0, max.(slopes, 0.0), color=(:red, 0.3))
  band!(ax1, dist_km, min.(slopes, 0.0), 0.0, color=(:green, 0.3))
  stairs!(ax1, dist_km, slopes, step=:post, color=:black, linewidth=1)

  # Curvature profile
  # Convert radius to 1/m curvature. A radius of 0 implies a straight track (infinite radius).
  curv_inv = [r == 0.0 ? 0.0 : 1.0 / r for r in curves]
  band!(ax2, dist_km, 0.0, curv_inv, color=(:red, 0.3))
  stairs!(ax2, dist_km, curv_inv, step=:post, color=:black, linewidth=1)

  # Speed profile
  lines!(ax3, dist_km, speeds_kmh, color=:black, linewidth=1.5)
  stairs!(ax3, dist_km, speed_limits_kmh, step=:post, color=(:red, 0.6), linestyle=(:dash, :dense), linewidth=1.0)

  # Energy profile
  lines!(ax4, dist_km, energy_MJ, color=:black, linewidth=1.5)

  # Time profile
  lines!(ax5, dist_km, times_min, color=:black, linewidth=1.5)


  # Link axes
  linkxaxes!(ax1, ax2, ax3, ax4, ax5, ax1_top, ax2_top, ax3_top, ax4_top, ax5_top)
  xlims!(ax1, dist_km[1], dist_km[end])

  # Station names
  xlabels = Vector{String}(undef, length(station_names))
  for i in 1:length(station_names)
    xlabels[i] = "$(station_names[i]) ($(round(station_pos_km[i], digits=2)))"
  end

  # Set station names at each X-axis so that grid lines align
  for ax_top in [ax1_top, ax2_top, ax3_top, ax4_top, ax5_top]
    ax_top.xticks = (station_pos_km, xlabels)
    ax_top.xticklabelrotation = π / 3
    hidespines!(ax_top)
    hideydecorations!(ax_top, grid=true) # also hides dummy y-grid
    hidexdecorations!(ax_top, grid=false) # also hides dummy y-grid
  end

  tot_stations = length(station_names)
  station_pos_km_tmp = copy(station_pos_km)
  station_pos_km_tmp[1] = station_pos_km[1] + 1 # helps show the first station name correctly
  text!(ax5, collect(zip(station_pos_km_tmp, fill(minimum(times_min), tot_stations))),
        text = xlabels, rotation=π/2, fontsize=10, font=:bold, color=:gray)
  
  for ax in [ax1, ax2, ax3, ax4, ax5]
    ax.xgridstyle = :dash
    ax.ygridstyle = :dash
  end

  # Hide distances from top plots, leave them at bottom plot only
  for ax in [ax1, ax2, ax3, ax4]
    hidexdecorations!(ax, grid=false)
  end

  # Hide stations from bottom plots, leave them at top plot only
  #for ax_top in [ax2_top, ax3_top, ax4_top, ax5_top]
  #  hidexdecorations!(ax_top, grid=false)
  #end

  # Title Header
  total_time = profile.time_min[end]
  minutes = floor(Int, total_time)
  seconds = round(Int, (total_time * 60) % 60)
  Label(fig[0, 1], "Train Type: $(train.name) | Running Time: $(minutes) min $(seconds) seg", font=:bold)
  rowgap!(fig.layout, 5)
  colsize!(fig.layout, 1, Fixed(1200))
  resize_to_layout!(fig)

  # Save figure
  if out_filepath !== nothing
    CairoMakie.activate!()
    save(out_filepath, fig, px_per_unit=2)
    println("Plot saved to: $out_filepath")
  end

  # Show plot
  if show
    GLMakie.activate!()
    screen = display(fig)
    wait(screen)
  end

  return fig
end

#if abspath(PROGRAM_FILE) == @__FILE__
function main()
  args = parse_commandline()

  geom_filepath = args["geometry"]
  train_filepath = args["train"]
  is_reversed = args["reverse"]
  ppkm = args["ppkm"]
  out_filepath = args["out"]
  show = args["show"]

  # TODO: implement dwelling
  dwell_time = args["dwell"]

  println("Loading track geometry data from: $geom_filepath")
  path = Path(geom_filepath, ppkm=ppkm, is_reversed=is_reversed)

  println("Loading train data from: $train_filepath")
  train = Train(train_filepath)

  println("Running simulation...")
  sim = TrainRunSimulation(train, path)
  profile = sim.profile

  total_time = profile.time_min[end]
  minutes = floor(Int, total_time)
  seconds = round(Int, (total_time * 60) % 60)
  total_energy = profile.energy_MJ[end]

  println("Simulation Complete: $(train.name)")
  println("Total Running Time: $(minutes) min $(seconds) sec")
  println("Total Energy Consumed: $(round(total_energy, digits=2)) MJ")

  plot_multi_profile(
    profile, 
    path, 
    train, 
    out_filepath=out_filepath,
    show=show
  )
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TrainRun.main()
end

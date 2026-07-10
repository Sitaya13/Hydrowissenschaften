import math
import matplotlib.pyplot as plt
import numpy as np

# Global variables
n = 51 # discretization along channel
u_new = [0.0] * n
u_old = [0.0] * n
error_tolerance = 1.0e-4

# File handles
out_file = None
out_python = None


def RUN_NewtonStep():
    """Perform one Newton step and return the error."""
    global u_new, u_old, out_python

    # Geometry
    x = [-100.0 + i * 10.0 for i in range(n)]
    bottom_elevation = [0.04 - i * 0.004 for i in range(n)]

    # Constants
    discharge = 0.1          # m3/s
    gravity = 9.81            # m/s2
    friction_law_exponent = 0.5
    bed_slope = 0.0004        # m/m
    bottom_width = 1.0        # m
    m = 1.0
    friction_coefficient = 10.0

    # Local arrays
    wetted_cross_section = [0.0] * n
    water_level_elevation = [0.0] * n
    flow_velocity = [0.0] * n
    Froude_number = [0.0] * n
    wetted_perimeter = [0.0] * n
    hydraulic_radius = [0.0] * n
    friction_slope = [0.0] * n

    # Debug output (overwrites each call)
    with open("out2.txt", "w") as out_file2:
        out_file2.write("Water depth (old):\t\t")
        for val in u_old:
            out_file2.write(f"\t{val:.3f}")
        out_file2.write("\n")

        for i in range(n):
            wetted_perimeter[i] = bottom_width + 2.0 * math.sqrt(1.0 + m * m) * u_old[i]
            wetted_cross_section[i] = (bottom_width + m * u_old[i]) * u_old[i]
            hydraulic_radius[i] = wetted_cross_section[i] / wetted_perimeter[i]
            water_level_elevation[i] = bottom_elevation[i] + u_old[i]
            flow_velocity[i] = discharge / wetted_cross_section[i]
            Froude_number[i] = flow_velocity[i] / math.sqrt(
                gravity * wetted_cross_section[i] /
                math.sqrt(bottom_width * bottom_width + 4.0 * m * wetted_cross_section[i])
            )
            friction_slope[i] = pow(
                flow_velocity[i] / (friction_coefficient * pow(hydraulic_radius[i], friction_law_exponent)),
                2
            )

        out_file2.write("Wetted perimeter:\t\t")
        for val in wetted_perimeter:
            out_file2.write(f"\t{val:.3f}")
        out_file2.write("\n")

    # Newton step (i = 0 .. n-2)
    for i in range(n - 1):
        N1 = pow(discharge, 2) / pow(wetted_cross_section[i + 1], 2) + gravity * u_old[i + 1]
        N2 = pow(discharge, 2) / pow(wetted_cross_section[i], 2) + gravity * u_old[i]
        N3 = gravity * (bed_slope - (friction_slope[i + 1] + friction_slope[i]) / 2.0) * (x[i + 1] - x[i])
        N = N1 - N2 - N3

        D1 = pow(discharge, 2) / pow(wetted_cross_section[i], 3) * (bottom_width + 2.0 * m * u_old[i]) - gravity
        D21 = friction_law_exponent * 2.0 * (math.sqrt(1 + m * m)) / wetted_perimeter[i]
        D22 = (1.0 + friction_law_exponent) / wetted_cross_section[i] * (bottom_width + 2.0 * m * u_old[i])
        D2 = gravity * friction_slope[i] * (D21 - D22) * (x[i + 1] - x[i])
        D = D1 + D2

        u_new[i] = u_old[i] - N / D

    # Compute error (sum of differences)
    error_sum = 0.0
    for i in range(n - 1):
        error_sum += u_old[i] - u_new[i]
    error = math.sqrt(error_sum * error_sum)

    # Update old values
    for i in range(n - 1):
        u_old[i] = u_new[i]

    # Write to CSV (appended)
    for i in range(n):
        out_python.write(f"{x[i]},{u_new[i]}\n")

    return error


def plot_all_profiles(x, profiles):
    """
    Plot all saved profiles (each a list of u values) in one figure.
    profiles: list of lists, each list is the water depth at a given iteration.
    """
    n_iter = len(profiles)
    if n_iter == 0:
        print("No profiles to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use a colormap to show progression from early (blue) to late (red)
    colors = plt.cm.viridis(np.linspace(0, 1, n_iter))

    for idx, u in enumerate(profiles):
        label = f"Iter {idx}" if idx < n_iter - 1 else f"Iter {idx} (final)"
        linewidth = 2.5 if idx == n_iter - 1 else 1.0
        alpha = 0.8 if idx == n_iter - 1 else 0.6
        ax.plot(x, u, color=colors[idx], linewidth=linewidth, alpha=alpha, label=label)
        ax.plot(x, u, color=colors[idx], linewidth=linewidth, alpha=alpha)

    ax.set_xlabel('Distance along channel (m)')
    ax.set_ylabel('Water depth (m)')
    ax.set_title('Evolution of Water Depth Profile over Newton Iterations')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_iter-1))
    sm.set_array([])

    # Add colorbar with explicit axes
    step = max(1, n_iter // 10)
    ticks = list(range(0, n_iter, step))
    if ticks[-1] != n_iter - 1:
        ticks.append(n_iter - 1)
    cbar = fig.colorbar(sm, ax=ax, label='Iteration number', ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])

    # Optionally add a legend (may be crowded, but keep it)
    ##ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig('all_profiles.png', dpi=150)
    plt.show()


def main():
    global u_old, u_new, out_file, out_python

    # Initialise
    for i in range(n):
        u_old[i] = 0.25
    u_old[n - 1] = 0.35
    u_new[n - 1] = 0.35

    # Open output files
    out_file = open("out.txt", "w")
    out_python = open("out.csv", "w")

    # x coordinates (for plotting)
    x = [-100.0 + i * 10.0 for i in range(n)]

    # List to store profiles after each Newton step
    profiles = []

    # Newton loop
    iteration = 0
    error = 1.1 * error_tolerance
    while error > error_tolerance:
        error = RUN_NewtonStep()
        print(f"Iteration {iteration}: error = {error:.6f}")

        # Save a copy of the current u_new
        profiles.append(u_new.copy())

        iteration += 1

    print(f"Converged in {iteration} iterations")

    # Write final water depths to out.txt
    out_file.write("Water depth (new):\t\t")
    for val in u_new:
        out_file.write(f"\t{val:.3f}")
    out_file.write("\n")

    out_file.close()
    out_python.close()

    # Plot all collected profiles in one figure
    plot_all_profiles(x, profiles)


if __name__ == "__main__":
    main()
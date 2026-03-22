#include <math.h>
#include <stdlib.h>
#include <emscripten.h>

#define G 9.81
#define KMH_TO_MS (1.0 / 3.6)
#define MS_TO_KMH 3.6

static double effective_mass(double mass, double rot_inertia) {
    return mass * (1.0 + 0.01 * rot_inertia);
}

static double rolling_resistance(double mass, double v, double davis_a, double davis_b, double davis_c) {
    double v_kmh = v * MS_TO_KMH;
    return mass * G * (davis_a + v_kmh * (davis_b + davis_c * v_kmh)) * 0.001;
}

static double slope_resistance(double mass, double slope) {
    return mass * G * slope * 0.001;
}

static double curve_resistance(double mass, double curve) {
    double curv_res = 0.0;
    double curv = fabs(curve);
    if (curv > 0) {
        double denom, specific_resistance;
        if (curv >= 300.0) {
            denom = fmax(1.0, curv - 55.0);
            specific_resistance = 650.0 / denom;
        } else {
            denom = fmax(1.0, curv - 30.0);
            specific_resistance = 500.0 / denom;
        }
        curv_res = mass * G * specific_resistance * 0.001;
    }
    return curv_res;
}

static double max_braking_effort(double v, double mass, double adh_mass, int n_brake, double* brake_v, double* brake_a) {
    double deceleration = brake_a[0];
    for (int i = 0; i < n_brake; i++) {
        if (v < brake_v[i]) {
            deceleration = brake_a[i];
            break;
        }
    }
    if (v >= brake_v[n_brake - 1]) deceleration = brake_a[n_brake - 1];

    double braking_force = mass * deceleration;
    double friction_coeff = 0.161 + 2.1 / (v + 12.2);
    double weather_coeff = 1.25;
    double adhesion_limit = adh_mass * G * friction_coeff * weather_coeff;
    return fmin(braking_force, adhesion_limit);
}

static double max_tractive_effort(double v, double mass, double adh_mass, int n_trac, double* trac_v, double* trac_f) {
    double traction_force = 0.0;
    if (v <= trac_v[0]) {
        traction_force = trac_f[0];
    } else if (v >= trac_v[n_trac - 1]) {
        traction_force = 0.0;
    } else {
        for (int i = 1; i < n_trac; i++) {
            if (v < trac_v[i]) {
                double v1 = trac_v[i - 1], f1 = trac_f[i - 1];
                double v2 = trac_v[i], f2 = trac_f[i];
                double p1 = f1 * v1, p2 = f2 * v2;
                double p = p1 + (p2 - p1) / (v2 - v1) * (v - v1);
                traction_force = p / v;
                break;
            }
        }
    }

    double friction_coeff = 0.161 + 2.1 / (v + 12.2);
    double weather_coeff = 1.25;
    double adhesion_limit = adh_mass * G * friction_coeff * weather_coeff;
    return fmin(traction_force, adhesion_limit);
}

// Main Simulation Entry Point
EMSCRIPTEN_KEEPALIVE
void run_simulation(
    int n,
    double* positions, double* speed_limits, double* slopes, double* curves,
    double* station_masks, double* dwell_times,
    double mass, double adh_mass, double rot_inertia,
    double davis_a, double davis_b, double davis_c,
    int n_trac, double* trac_v, double* trac_f,
    int n_brake, double* brake_v, double* brake_a,
    double* out_v, double* out_time, double* out_energy, double* out_force
) {
    double* v_fwd = (double*)malloc(n * sizeof(double));
    double* v_bwd = (double*)malloc(n * sizeof(double));
    double v_train_lim = trac_v[n_trac - 1];

    // Forward Pass
    v_fwd[0] = 0.0;
    for (int i = 0; i < n - 1; i++) {
        double v_curr = v_fwd[i];
        double F_t = max_tractive_effort(v_curr, mass, adh_mass, n_trac, trac_v, trac_f);
        double R_roll = rolling_resistance(mass, v_curr, davis_a, davis_b, davis_c);
        double R_slope = slope_resistance(mass, slopes[i]);
        double R_curve = curve_resistance(mass, curves[i]);

        double F_net = F_t - R_roll - R_slope - R_curve;
        double a = F_net / effective_mass(mass, rot_inertia);
        
        double ds = positions[i + 1] - positions[i];
        double v_next_sq = v_curr * v_curr + 2 * a * ds;
        double v_next = (v_next_sq > 0) ? sqrt(v_next_sq) : 0.0;
        double v_track_lim = station_masks[i + 1] > 0.5 ? 0.0 : speed_limits[i + 1];

        v_fwd[i + 1] = fmin(fmin(v_next, v_track_lim), v_train_lim);
    }

    // Backward Pass
    v_bwd[n - 1] = 0.0;
    for (int i = n - 1; i > 0; i--) {
        double v_curr = v_bwd[i];
        double F_b = max_braking_effort(v_curr, mass, adh_mass, n_brake, brake_v, brake_a);
        double R_roll = rolling_resistance(mass, v_curr, davis_a, davis_b, davis_c);
        double R_slope = slope_resistance(mass, slopes[i]);
        double R_curve = curve_resistance(mass, curves[i]);

        double F_net = F_b + R_roll - R_slope + R_curve;
        double a = F_net / effective_mass(mass, rot_inertia);

        double ds = positions[i] - positions[i - 1];
        double v_prev_sq = v_curr * v_curr + 2 * a * ds;
        double v_prev = (v_prev_sq > 0) ? sqrt(v_prev_sq) : 0.0;
        double v_track_lim = station_masks[i - 1] > 0.5 ? 0.0 : speed_limits[i - 1];

        v_bwd[i - 1] = fmin(fmin(v_prev, v_track_lim), v_train_lim);
    }

    // Merge & Accumulate
    out_time[0] = 0.0;
    out_energy[0] = 0.0;
    out_v[0] = fmin(v_fwd[0], v_bwd[0]);

    for (int i = 0; i < n - 1; i++) {
        double v1 = out_v[i];
        double v2 = fmin(v_fwd[i + 1], v_bwd[i + 1]);
        out_v[i + 1] = v2;

        double v_avg = (v1 + v2) / 2.0;
        double ds = positions[i + 1] - positions[i];
        
        double dt = (v_avg > 0) ? (ds / v_avg) : 0.0;
        out_time[i + 1] = out_time[i] + dt + dwell_times[i];

        double acc = (ds > 0) ? (v2 * v2 - v1 * v1) / (2 * ds) : 0.0;
        double R_roll = rolling_resistance(mass, v_avg, davis_a, davis_b, davis_c);
        double R_slope = slope_resistance(mass, slopes[i]);
        double R_curve = curve_resistance(mass, curves[i]);

        double force = effective_mass(mass, rot_inertia) * acc + R_roll + R_slope + R_curve;

        if (force >= 0) {
            out_force[i] = fmin(force, max_tractive_effort(v_avg, mass, adh_mass, n_trac, trac_v, trac_f));
            out_energy[i + 1] = out_energy[i] + (out_force[i] * ds);
        } else {
            out_force[i] = fmax(force, -max_braking_effort(v_avg, mass, adh_mass, n_brake, brake_v, brake_a));
            out_energy[i + 1] = out_energy[i];
        }
    }
    
    out_time[n - 1] += dwell_times[n - 1];
    out_force[n - 1] = 0.0;

    free(v_fwd);
    free(v_bwd);
}

/* The smallest useful program:  make && ./example
 *
 * Runs the model open loop and the policy on each column's typical value
 * (its mean in the training data, from the generated headers), and times
 * them. Real inputs are windows of your own measurements, in physical units.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>

#ifdef NT_HAVE_MODEL
#include "nt_model.h"
#endif
#ifdef NT_HAVE_POLICY
#include "nt_policy.h"
#endif

#define STEPS 1000

static double seconds(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

#ifdef NT_HAVE_MODEL
static void run_model(void)
{
    float y[NT_MODEL_HISTORY][NT_MODEL_NY], u[NT_MODEL_HISTORY][NT_MODEL_NU];
    float d[NT_MODEL_HISTORY][NT_MODEL_ND];
    nt_model_state state;
    double start;

    for (int r = 0; r < NT_MODEL_HISTORY; r++) {
        for (int i = 0; i < NT_MODEL_NY; i++) y[r][i] = NT_MODEL_Y_TYPICAL[i];
        for (int i = 0; i < NT_MODEL_NU; i++) u[r][i] = NT_MODEL_U_TYPICAL[i];
        for (int i = 0; i < NT_MODEL_ND; i++) d[r][i] = NT_MODEL_D_TYPICAL[i];
    }
    nt_model_warmup(&state, y, u, d);

    /* A controller would choose u each step; here it holds still. */
    start = seconds();
    for (int k = 0; k < STEPS; k++) nt_model_step(&state, NT_MODEL_U_TYPICAL, NT_MODEL_D_TYPICAL);
    printf("model: %d open-loop steps, %.2f us per step\n", STEPS, (seconds() - start) / STEPS * 1e6);
    for (int i = 0; i < NT_MODEL_NY; i++) {
        printf("  %s = %.4g after %d steps\n", NT_MODEL_Y_COLUMNS[i], state.outputs[i], STEPS);
    }
}
#endif

#ifdef NT_HAVE_POLICY
static void run_policy(void)
{
    float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N];
#if NT_POLICY_HAS_D
    float d[NT_POLICY_D_STEPS][NT_POLICY_D_N];
#endif
#if NT_POLICY_HAS_U_HISTORY
    float u_history[NT_POLICY_U_HISTORY_STEPS][NT_POLICY_U_HISTORY_N];
#endif
#if NT_POLICY_HAS_R
    float r[NT_POLICY_R_STEPS][NT_POLICY_R_N];
#endif
    float u[NT_POLICY_NU];
    double start;

    for (int t = 0; t < NT_POLICY_Y_STEPS; t++)
        for (int i = 0; i < NT_POLICY_Y_N; i++) y[t][i] = NT_POLICY_Y_TYPICAL[i];
#if NT_POLICY_HAS_U_HISTORY
    for (int t = 0; t < NT_POLICY_U_HISTORY_STEPS; t++)
        for (int i = 0; i < NT_POLICY_U_HISTORY_N; i++) u_history[t][i] = NT_POLICY_U_HISTORY_TYPICAL[i];
#endif
#if NT_POLICY_HAS_D
    for (int t = 0; t < NT_POLICY_D_STEPS; t++)
        for (int i = 0; i < NT_POLICY_D_N; i++) d[t][i] = NT_POLICY_D_TYPICAL[i];
#endif
#if NT_POLICY_HAS_R
    for (int t = 0; t < NT_POLICY_R_STEPS; t++)
        for (int i = 0; i < NT_POLICY_R_N; i++) r[t][i] = NT_POLICY_R_TYPICAL[i];
#endif

    start = seconds();
    for (int k = 0; k < STEPS; k++) {
        nt_policy_act(y,
#if NT_POLICY_HAS_U_HISTORY
                      u_history,
#endif
#if NT_POLICY_HAS_D
                      d,
#endif
#if NT_POLICY_HAS_R
                      r,
#endif
                      u);
    }
    printf("policy: %d commands, %.2f us per command\n", STEPS, (seconds() - start) / STEPS * 1e6);
    for (int i = 0; i < NT_POLICY_NU; i++) printf("  %s = %.4g\n", NT_POLICY_U_COLUMNS[i], u[i]);
}
#endif

int main(void)
{
#ifdef NT_HAVE_MODEL
    run_model();
#endif
#ifdef NT_HAVE_POLICY
    run_policy();
#endif
    return 0;
}

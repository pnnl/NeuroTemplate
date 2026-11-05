#include "nt_policy.h"

#include <string.h>

const char *const nt_policy_export_id = NT_POLICY_EXPORT_ID;

void nt_policy_act(const float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N],
#if NT_POLICY_HAS_U_HISTORY
                   const float u_history[NT_POLICY_U_HISTORY_STEPS][NT_POLICY_U_HISTORY_N],
#endif
#if NT_POLICY_HAS_D
                   const float d[NT_POLICY_D_STEPS][NT_POLICY_D_N],
#endif
#if NT_POLICY_HAS_R
                   const float r[NT_POLICY_R_STEPS][NT_POLICY_R_N],
#endif
                   float u[NT_POLICY_NU])
{
    float out[1][NT_POLICY_NU];
    /* The graph takes a batch of one: each window as [1][steps][n]. */
    nt_policy_graph((const float (*)[NT_POLICY_Y_STEPS][NT_POLICY_Y_N])y,
#if NT_POLICY_HAS_U_HISTORY
                    (const float (*)[NT_POLICY_U_HISTORY_STEPS][NT_POLICY_U_HISTORY_N])u_history,
#endif
#if NT_POLICY_HAS_D
                    (const float (*)[NT_POLICY_D_STEPS][NT_POLICY_D_N])d,
#endif
#if NT_POLICY_HAS_R
                    (const float (*)[NT_POLICY_R_STEPS][NT_POLICY_R_N])r,
#endif
                    out);
    memcpy(u, out[0], sizeof out[0]);
}

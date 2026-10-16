#include "nt_policy.h"

#include <string.h>

const char *const nt_policy_export_id = NT_POLICY_EXPORT_ID;

typedef const float (*y_window)[NT_POLICY_Y_STEPS][NT_POLICY_Y_N];
typedef const float (*d_window)[NT_POLICY_D_STEPS][NT_POLICY_D_N];

#if NT_POLICY_HAS_R
typedef const float (*r_window)[NT_POLICY_R_STEPS][NT_POLICY_R_N];

void nt_policy_act(const float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N],
                   const float d[NT_POLICY_D_STEPS][NT_POLICY_D_N],
                   const float r[NT_POLICY_R_STEPS][NT_POLICY_R_N],
                   float u[NT_POLICY_NU])
{
    float command[1][NT_POLICY_NU];
    nt_policy_graph((y_window)y, (d_window)d, (r_window)r, command);
    memcpy(u, command[0], sizeof command[0]);
}
#else
void nt_policy_act(const float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N],
                   const float d[NT_POLICY_D_STEPS][NT_POLICY_D_N],
                   float u[NT_POLICY_NU])
{
    float command[1][NT_POLICY_NU];
    nt_policy_graph((y_window)y, (d_window)d, command);
    memcpy(u, command[0], sizeof command[0]);
}
#endif

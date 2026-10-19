/* Run an exported NeuroTemplate control policy from C.
 *
 * The C twin of runtime.py's Policy, shipped verbatim in every C export.
 * Stateless: one call per control step. Each input is a window of physical
 * values, oldest row first, whose length, columns and current row
 * policy/policy.h states (NT_POLICY_Y_STEPS, NT_POLICY_Y_COLUMNS,
 * NT_POLICY_Y_CURRENT_ROW, ...). `y` and `d` end at the current step; `d`
 * and `r` may run past it into the known future. `u_history`, only when
 * NT_POLICY_HAS_U_HISTORY, holds the commands applied over the last
 * NT_POLICY_U_HISTORY_STEPS steps and ends the step before the current one.
 */
#ifndef NT_POLICY_H
#define NT_POLICY_H

#include "policy/policy.h"

/* The command for one step, into `u` (NT_POLICY_U_COLUMNS order). */
void nt_policy_act(const float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N],
#if NT_POLICY_HAS_U_HISTORY
                   const float u_history[NT_POLICY_U_HISTORY_STEPS][NT_POLICY_U_HISTORY_N],
#endif
                   const float d[NT_POLICY_D_STEPS][NT_POLICY_D_N],
#if NT_POLICY_HAS_R
                   const float r[NT_POLICY_R_STEPS][NT_POLICY_R_N],
#endif
                   float u[NT_POLICY_NU]);

/* Which export this graph came from; policy/policy.json says the same. */
extern const char *const nt_policy_export_id;

#endif

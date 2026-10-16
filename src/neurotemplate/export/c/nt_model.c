#include "nt_model.h"

#include <string.h>

const char *const nt_model_export_id = NT_MODEL_EXPORT_ID;

void nt_model_warmup(nt_model_state *state,
                     const float y[NT_MODEL_HISTORY][NT_MODEL_NY],
                     const float u[NT_MODEL_HISTORY][NT_MODEL_NU],
                     const float d[NT_MODEL_HISTORY][NT_MODEL_ND])
{
    memcpy(state->y, y, sizeof state->y);
    memcpy(state->u, u, sizeof state->u);
    memcpy(state->d, d, sizeof state->d);
    memcpy(state->outputs, y[NT_MODEL_HISTORY - 1], sizeof state->outputs);
    state->warm = 0;
}

void nt_model_step(nt_model_state *state, const float u_now[NT_MODEL_NU],
                   const float d_now[NT_MODEL_ND])
{
    typedef const float (*y_window)[NT_MODEL_HISTORY][NT_MODEL_NY];
    typedef const float (*u_window)[NT_MODEL_HISTORY][NT_MODEL_NU];
    typedef const float (*d_window)[NT_MODEL_HISTORY][NT_MODEL_ND];
    float y_next[1][NT_MODEL_NY];
    int row;

    memcpy(state->u[NT_MODEL_HISTORY - 1], u_now, sizeof state->u[0]);
    memcpy(state->d[NT_MODEL_HISTORY - 1], d_now, sizeof state->d[0]);

    /* The first step seeds the hidden state from the window; every later
     * one carries it explicitly. */
#if NT_MODEL_STATE > 0
    float state_next[1][NT_MODEL_STATE];
    if (state->warm) {
        nt_step_graph((y_window)state->y, (u_window)state->u, (d_window)state->d,
                      (const float (*)[NT_MODEL_STATE])state->hidden, y_next, state_next);
    } else {
        nt_warm_start_graph((y_window)state->y, (u_window)state->u, (d_window)state->d,
                            y_next, state_next);
    }
    memcpy(state->hidden, state_next[0], sizeof state_next[0]);
#else
    if (state->warm) {
        nt_step_graph((y_window)state->y, (u_window)state->u, (d_window)state->d, y_next);
    } else {
        nt_warm_start_graph((y_window)state->y, (u_window)state->u, (d_window)state->d, y_next);
    }
#endif

    /* Slide every window one row: the prediction becomes the newest output. */
    for (row = 0; row < NT_MODEL_HISTORY - 1; row++) {
        memcpy(state->y[row], state->y[row + 1], sizeof state->y[0]);
        memcpy(state->u[row], state->u[row + 1], sizeof state->u[0]);
        memcpy(state->d[row], state->d[row + 1], sizeof state->d[0]);
    }
    memcpy(state->y[NT_MODEL_HISTORY - 1], y_next[0], sizeof y_next[0]);
    memcpy(state->outputs, y_next[0], sizeof y_next[0]);
    state->warm = 1;
}

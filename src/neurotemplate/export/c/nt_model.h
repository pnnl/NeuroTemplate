/* Run an exported NeuroTemplate dynamics model from C.
 *
 * The C twin of runtime.py's Model, shipped verbatim in every C export. It
 * holds no numeric logic: normalisation, the network and the hidden state's
 * seeding are inside the generated graphs (model/warm_start.c, model/step.c).
 * This file only moves windows of physical values in and out of them. No
 * heap, no I/O: the whole model state is one fixed-size struct.
 *
 * Every array is in physical units, its columns in the order model/model.h
 * lists them (NT_MODEL_Y_COLUMNS, ...), windows oldest row first.
 */
#ifndef NT_MODEL_H
#define NT_MODEL_H

#include "model/model.h"

/* Everything the model needs to take one more step. `outputs` is the latest
 * prediction, or straight after nt_model_warmup the last measured row. The
 * rest is the model's memory; treat it as opaque. */
typedef struct {
    float outputs[NT_MODEL_NY];
    float y[NT_MODEL_HISTORY][NT_MODEL_NY];
    float u[NT_MODEL_HISTORY][NT_MODEL_NU];  /* the last row is this step's, set by nt_model_step */
    float d[NT_MODEL_HISTORY][NT_MODEL_ND];
    float hidden[NT_MODEL_STATE_SLOTS];      /* NT_MODEL_STATE values; unused when 0 */
    int warm;                                /* has the first step run? */
} nt_model_state;

/* Start from the last NT_MODEL_HISTORY rows of recent measurements. The
 * final u and d rows are not used: they are the inputs to the first step. */
void nt_model_warmup(nt_model_state *state,
                     const float y[NT_MODEL_HISTORY][NT_MODEL_NY],
                     const float u[NT_MODEL_HISTORY][NT_MODEL_NU],
                     const float d[NT_MODEL_HISTORY][NT_MODEL_ND]);

/* Advance one timestep under this step's inputs; state->outputs then holds
 * the prediction one timestep later. */
void nt_model_step(nt_model_state *state, const float u_now[NT_MODEL_NU],
                   const float d_now[NT_MODEL_ND]);

/* Which export these graphs came from; model/model.json says the same. */
extern const char *const nt_model_export_id;

#endif

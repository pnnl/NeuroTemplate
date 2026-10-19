/* Check that this package works from C on this machine:  make && ./validate
 *
 * The C twin of validate.py. It reproduces each artifact's reference/ fixture,
 * which the *trained* model wrote at export time, through the C runtime and this
 * machine's compiler. Passing means the whole chain reproduces the model:
 * trained model -> export -> generated C -> this platform. Exits non-zero
 * if any check fails.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NT_HAVE_MODEL
#include "nt_model.h"
#endif
#ifdef NT_HAVE_POLICY
#include "nt_policy.h"
#endif

/* A CSV as every shipped fixture is: a header, then an index column and
 * numbers. Quoted fields are refused rather than mis-split. */
typedef struct {
    int columns;     /* excluding the index */
    int rows;
    char **names;    /* [columns] */
    double *values;  /* [rows][columns] */
} table;

static char *read_line(FILE *file)
{
    size_t size = 256, length = 0;
    char *line = malloc(size);
    int c;
    while ((c = fgetc(file)) != EOF && c != '\n') {
        if (length + 1 >= size) line = realloc(line, size *= 2);
        line[length++] = (char)c;
    }
    if (c == EOF && length == 0) { free(line); return NULL; }
    if (length && line[length - 1] == '\r') length--;
    line[length] = '\0';
    if (strchr(line, '"')) {
        fprintf(stderr, "quoted CSV fields are not supported: %s\n", line);
        exit(2);
    }
    return line;
}

static table read_csv(const char *path)
{
    table t = {0, 0, NULL, NULL};
    FILE *file = fopen(path, "r");
    char *line, *field;
    size_t capacity = 0;
    if (!file) { fprintf(stderr, "cannot open %s\n", path); exit(2); }

    line = read_line(file);
    for (field = strchr(line, ','); field; field = strchr(field + 1, ',')) t.columns++;
    t.names = malloc(t.columns * sizeof *t.names);
    strtok(line, ",");                       /* the index's name */
    for (int i = 0; i < t.columns; i++) t.names[i] = strdup(strtok(NULL, ","));
    free(line);

    while ((line = read_line(file))) {
        if (!*line) { free(line); continue; }
        if ((size_t)(t.rows + 1) * t.columns > capacity) {
            capacity = capacity ? capacity * 2 : (size_t)t.columns * 64;
            t.values = realloc(t.values, capacity * sizeof *t.values);
        }
        strtok(line, ",");                   /* the index */
        for (int i = 0; i < t.columns; i++) {
            char *number = strtok(NULL, ","), *end;
            t.values[(size_t)t.rows * t.columns + i] = number ? strtod(number, &end) : NAN;
            if (!number || end == number) {
                fprintf(stderr, "%s: row %d has a missing or non-numeric field\n", path, t.rows + 1);
                exit(2);
            }
        }
        t.rows++;
        free(line);
    }
    fclose(file);
    return t;
}

/* Where `name` sits among the table's columns; exits if it is absent. */
static int column(const table *t, const char *name, const char *path)
{
    for (int i = 0; i < t->columns; i++) {
        if (strcmp(t->names[i], name) == 0) return i;
    }
    fprintf(stderr, "%s has no column %s\n", path, name);
    exit(2);
}

static double cell(const table *t, int row, int col)
{
    return t->values[(size_t)row * t->columns + col];
}

static int failures = 0, checked = 0;

static void check(const char *name, int ok)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    checked++;
    if (!ok) failures++;
}

#if defined(NT_HAVE_MODEL) && NT_MODEL_VERIFIED
static void check_model(void)
{
    const char *input_path = "model/reference/example_input.csv";
    const char *output_path = "model/reference/expected_output.csv";
    table input = read_csv(input_path), expected = read_csv(output_path);
    float y[NT_MODEL_HISTORY][NT_MODEL_NY], u[NT_MODEL_HISTORY][NT_MODEL_NU];
    float d[NT_MODEL_HISTORY][NT_MODEL_ND], u_now[NT_MODEL_NU], d_now[NT_MODEL_ND];
    int y_col[NT_MODEL_NY], u_col[NT_MODEL_NU], d_col[NT_MODEL_ND], out_col[NT_MODEL_NY];
    double error[NT_MODEL_NY] = {0}, state_error = 0.0;
    nt_model_state state;
    int ok = 1;

    for (int i = 0; i < NT_MODEL_NY; i++) {
        y_col[i] = column(&input, NT_MODEL_Y_COLUMNS[i], input_path);
        out_col[i] = column(&expected, NT_MODEL_Y_COLUMNS[i], output_path);
    }
    for (int i = 0; i < NT_MODEL_NU; i++) u_col[i] = column(&input, NT_MODEL_U_COLUMNS[i], input_path);
    for (int i = 0; i < NT_MODEL_ND; i++) d_col[i] = column(&input, NT_MODEL_D_COLUMNS[i], input_path);

    for (int r = 0; r < NT_MODEL_HISTORY; r++) {
        for (int i = 0; i < NT_MODEL_NY; i++) y[r][i] = (float)cell(&input, r, y_col[i]);
        for (int i = 0; i < NT_MODEL_NU; i++) u[r][i] = (float)cell(&input, r, u_col[i]);
        for (int i = 0; i < NT_MODEL_ND; i++) d[r][i] = (float)cell(&input, r, d_col[i]);
    }
    nt_model_warmup(&state, y, u, d);

#if NT_MODEL_STATE > 0
    table expected_state = read_csv("model/reference/expected_state.csv");
#endif
    /* Step k predicts row HISTORY + k from the inputs in the row before it. */
    for (int k = 0; k < expected.rows; k++) {
        int row = NT_MODEL_HISTORY - 1 + k;
        for (int i = 0; i < NT_MODEL_NU; i++) u_now[i] = (float)cell(&input, row, u_col[i]);
        for (int i = 0; i < NT_MODEL_ND; i++) d_now[i] = (float)cell(&input, row, d_col[i]);
        nt_model_step(&state, u_now, d_now);
        for (int i = 0; i < NT_MODEL_NY; i++) {
            double e = fabs((double)state.outputs[i] - cell(&expected, k, out_col[i]));
            if (e > error[i]) error[i] = e;
        }
#if NT_MODEL_STATE > 0
        for (int i = 0; i < NT_MODEL_STATE; i++) {
            double e = fabs((double)state.hidden[i] - cell(&expected_state, k, i));
            if (e > state_error) state_error = e;
        }
#endif
    }

    for (int i = 0; i < NT_MODEL_NY; i++) {
        printf("  model: %s: max |difference| %.2e (tolerance %.0e)\n",
               NT_MODEL_Y_COLUMNS[i], error[i], NT_MODEL_Y_TOLERANCE[i]);
        if (error[i] > NT_MODEL_Y_TOLERANCE[i]) ok = 0;
    }
    printf("  model: hidden state max |difference| %.2e (tolerance %.0e)\n",
           state_error, NT_MODEL_STATE_TOLERANCE);
    if (state_error > NT_MODEL_STATE_TOLERANCE) {
        ok = 0;
        printf("  the hidden state disagrees: warm-up or step is wrong, not just the output layer\n");
    }
    check("model matches model/reference/", ok);
}
#endif

#ifdef NT_HAVE_POLICY
static void check_policy(void)
{
    /* inputs.csv holds one window per row, time-major: y, u_history, d, then r. */
    table input = read_csv("policy/reference/inputs.csv");
    table expected = read_csv("policy/reference/expected_actions.csv");
    float y[NT_POLICY_Y_STEPS][NT_POLICY_Y_N], d[NT_POLICY_D_STEPS][NT_POLICY_D_N];
#if NT_POLICY_HAS_U_HISTORY
    float u_history[NT_POLICY_U_HISTORY_STEPS][NT_POLICY_U_HISTORY_N];
#endif
#if NT_POLICY_HAS_R
    float r[NT_POLICY_R_STEPS][NT_POLICY_R_N];
#endif
    float u[NT_POLICY_NU];
    double error[NT_POLICY_NU] = {0};
    int ok = 1;

    for (int w = 0; w < input.rows; w++) {
        int c = 0;
        for (int t = 0; t < NT_POLICY_Y_STEPS; t++)
            for (int i = 0; i < NT_POLICY_Y_N; i++) y[t][i] = (float)cell(&input, w, c++);
#if NT_POLICY_HAS_U_HISTORY
        for (int t = 0; t < NT_POLICY_U_HISTORY_STEPS; t++)
            for (int i = 0; i < NT_POLICY_U_HISTORY_N; i++) u_history[t][i] = (float)cell(&input, w, c++);
#endif
        for (int t = 0; t < NT_POLICY_D_STEPS; t++)
            for (int i = 0; i < NT_POLICY_D_N; i++) d[t][i] = (float)cell(&input, w, c++);
#if NT_POLICY_HAS_R
        for (int t = 0; t < NT_POLICY_R_STEPS; t++)
            for (int i = 0; i < NT_POLICY_R_N; i++) r[t][i] = (float)cell(&input, w, c++);
#endif
        nt_policy_act(y,
#if NT_POLICY_HAS_U_HISTORY
                      u_history,
#endif
                      d,
#if NT_POLICY_HAS_R
                      r,
#endif
                      u);
        for (int i = 0; i < NT_POLICY_NU; i++) {
            double e = fabs((double)u[i] - cell(&expected, w, i));
            if (e > error[i]) error[i] = e;
        }
    }
    for (int i = 0; i < NT_POLICY_NU; i++) {
        printf("  policy: %s: max |difference| %.2e (tolerance %.0e)\n",
               NT_POLICY_U_COLUMNS[i], error[i], NT_POLICY_U_TOLERANCE[i]);
        if (error[i] > NT_POLICY_U_TOLERANCE[i]) ok = 0;
    }
    check("policy matches policy/reference/", ok);
}
#endif

int main(void)
{
    printf("reproducing the trained model's own answers\n");
#if defined(NT_HAVE_MODEL) && NT_MODEL_VERIFIED
    check_model();
#endif
#ifdef NT_HAVE_POLICY
    check_policy();
#endif
    if (failures) {
        printf("\nFAILED %d check(s)\n", failures);
        return 1;
    }
    printf("\nAll %d check(s) passed.\n", checked);
    return 0;
}

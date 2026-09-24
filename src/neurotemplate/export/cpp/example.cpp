// The smallest useful C++ program. From the package root:  build/example
//
// The C++ twin of example.py: a forecast and a step-by-step rollout with the
// model, and one command from the policy, all on the shipped reference data.

#include "runtime.hpp"

#include <cmath>
#include <cstdio>

// Row `i` of a row-major `[rows, n]` array.
static std::vector<float> row(const std::vector<float>& values, size_t n, size_t i) {
    return {values.begin() + i * n, values.begin() + (i + 1) * n};
}

// The first `count` rows of a row-major `[rows, n]` array.
static std::vector<float> head(const std::vector<float>& values, size_t n, size_t count) {
    return {values.begin(), values.begin() + count * n};
}

static void run_model() {
    nt::Model model("model");
    const size_t s = model.history_steps, ny = model.ny();
    std::printf("model: %zu outputs, %zu controls, %zu disturbances; warm-up %zu steps "
                "of %g s; hidden state width %d\n", ny, model.nu(), model.nd(), s,
                model.timestep_seconds, model.state_width);

    // Physical units throughout; the graphs normalise internally.
    nt::Table data = nt::read_csv("model/reference/example_input.csv");
    std::vector<float> y = data.select(model.y_columns);
    std::vector<float> u = data.select(model.u_columns);
    std::vector<float> d = data.select(model.d_columns);
    std::printf("loaded %zu rows, %s .. %s\n", data.rows.size(), data.index.front().c_str(),
                data.index.back().c_str());

    // 1. A forecast of the whole window. The first `history_steps` rows are
    // the warm-up; after that the model feeds on its own predictions and
    // reads only u and d.
    nt::Rollout forecast = model.rollout(y, u, d);
    const size_t steps = forecast.outputs.size() / ny;
    for (size_t i = 0; i < ny && i < 4; ++i) {
        double squared = 0.0;
        for (size_t k = 0; k < steps; ++k) {
            double error = forecast.outputs[k * ny + i] - y[(s + k) * ny + i];
            squared += error * error;
        }
        std::printf("  %s: open-loop RMSE %.3f against the recorded values\n",
                    model.y_columns[i].c_str(), std::sqrt(squared / steps));
    }

    // 2. Step by step. `step` returns a new State and never modifies the one
    // passed in. A controller chooses the u values itself.
    nt::State state = model.warmup(head(y, ny, s), head(u, model.nu(), s), head(d, model.nd(), s));
    for (size_t k = 0; k < 12; ++k) {
        state = model.step(state, row(u, model.nu(), s + k - 1), row(d, model.nd(), s + k - 1));
    }
    std::printf("12 steps later: %s = %.3f (recorded %.3f)\n", model.y_columns[0].c_str(),
                state.outputs[0], y[(s + 11) * ny]);
}

static void run_policy() {
    nt::Policy policy("policy");
    std::printf("\npolicy: %zu commands from", policy.u_columns.size());
    for (const nt::PolicyInput& input : policy.inputs) {
        std::printf(" %s %dx%zu", input.name.c_str(), input.steps, input.columns.size());
    }
    std::printf("\n");

    // The shipped reference holds real windows from a closed-loop rollout.
    auto windows = nt::read_windows("policy/reference/inputs.csv", policy.inputs);
    std::map<std::string, std::vector<float>> first;
    for (const nt::PolicyInput& input : policy.inputs) {
        first[input.name] = head(windows[input.name], input.steps * input.columns.size(), 1);
    }
    std::vector<float> command = policy.act(first);
    std::printf("  first reference window ->");
    for (size_t i = 0; i < command.size(); ++i) {
        std::printf(" %s=%.3f", policy.u_columns[i].c_str(), command[i]);
    }
    std::printf("\n");
}

int main() {
    try {
        if (nt::fs::is_directory("model")) run_model();
        if (nt::fs::is_directory("policy")) run_policy();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "example: %s\n", error.what());
        return 1;
    }
    return 0;
}

// Check that this package works from C++ on this machine:
//
//     build/validate [package_dir]      (default: the current directory)
//
// The C++ twin of validate.py. It reproduces every */reference/ fixture,
// which the *trained* model wrote at export time, through runtime.hpp and
// this machine's onnxruntime. Passing means the whole chain reproduces the
// model: trained model -> export -> this runtime -> this platform. Exits
// non-zero if any check fails.

#include "runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace {

// This machine's answers against the trained model's own, per column.
struct Check {
    std::string name;   // "model" or "policy"
    std::map<std::string, double> error, tolerance;
    double state_error = 0.0, state_tolerance = 0.0;

    bool ok() const {
        for (const auto& [column, err] : error) {
            if (err > tolerance.at(column)) return false;
        }
        return state_error <= state_tolerance;
    }

    void report() const {
        std::string worst;
        double worst_ratio = -1.0;
        for (const auto& [column, err] : error) {
            if (err / tolerance.at(column) > worst_ratio) {
                worst_ratio = err / tolerance.at(column);
                worst = column;
            }
        }
        std::printf("  %s: worst column %s: %.2e (tolerance %.0e)\n", name.c_str(),
                    worst.c_str(), error.at(worst), tolerance.at(worst));
        if (name == "model") {
            std::printf("  %s: hidden state max |difference| %.2e (tolerance %.0e)\n",
                        name.c_str(), state_error, state_tolerance);
        }
        for (const auto& [column, err] : error) {
            if (err > tolerance.at(column)) {
                std::printf("  %s: out of tolerance: %s\n", name.c_str(), column.c_str());
            }
        }
        if (state_error > state_tolerance) {
            std::printf("  the hidden state disagrees: warm-up or step is wrong, "
                        "not just the output layer\n");
        }
    }
};

// Worst |got - expected| per column. `got` is `[rows, columns.size()]`;
// `expected` is read as doubles, and compared in double as validate.py does.
std::map<std::string, double> column_errors(const std::vector<float>& got, const nt::Table& expected,
                                            const std::vector<std::string>& columns) {
    std::map<std::string, double> error;
    for (size_t i = 0; i < columns.size(); ++i) {
        size_t position = expected.position(columns[i]);
        double worst = 0.0;
        for (size_t r = 0; r < expected.rows.size(); ++r) {
            worst = std::max(worst, std::abs(double(got[r * columns.size() + i]) -
                                             expected.rows[r][position]));
        }
        error[columns[i]] = worst;
    }
    return error;
}

Check check_model(const nt::fs::path& model_dir) {
    const nt::fs::path reference = model_dir / "reference";
    nt::Json document = nt::read_json(reference / "REFERENCE.json");
    nt::Model model(model_dir);
    nt::Table frame = nt::read_csv(reference / "example_input.csv");
    nt::Table expected = nt::read_csv(reference / "expected_output.csv");
    const int steps = static_cast<int>(expected.rows.size());
    nt::Rollout rollout = model.rollout(frame.select(model.y_columns), frame.select(model.u_columns),
                                        frame.select(model.d_columns), steps);

    Check check{"model", column_errors(rollout.outputs, expected, model.y_columns),
                document.at("tolerance").at("y").get<std::map<std::string, double>>()};
    check.state_tolerance = document.at("tolerance").at("state").get<double>();
    if (model.state_width) {
        nt::Table state = nt::read_csv(reference / "expected_state.csv");
        std::map<std::string, double> errors = column_errors(rollout.hidden, state, state.columns);
        for (const auto& [column, err] : errors) check.state_error = std::max(check.state_error, err);
    }
    return check;
}

Check check_policy(const nt::fs::path& policy_dir) {
    const nt::fs::path reference = policy_dir / "reference";
    nt::Json document = nt::read_json(reference / "REFERENCE.json");
    nt::Policy policy(policy_dir);
    nt::Table expected = nt::read_csv(reference / "expected_actions.csv");
    auto windows = nt::read_windows(reference / "inputs.csv", policy.inputs);
    std::vector<float> actions = policy.act_batch(windows, static_cast<int64_t>(expected.rows.size()));
    return Check{"policy", column_errors(actions, expected, policy.u_columns),
                 document.at("tolerance").at("u").get<std::map<std::string, double>>()};
}

}  // namespace

int main(int argc, char** argv) {
    const nt::fs::path root = argc > 1 ? argv[1] : ".";
    std::vector<Check> checks;
    try {
        if (nt::fs::is_directory(root / "model")) checks.push_back(check_model(root / "model"));
        if (nt::fs::is_directory(root / "policy")) checks.push_back(check_policy(root / "policy"));
    } catch (const std::exception& error) {
        std::cerr << "validate: " << error.what() << "\n";
        return 2;
    }
    if (checks.empty()) {
        std::cerr << root << " holds neither model/ nor policy/.\n";
        return 2;
    }

    std::cout << "reproducing the trained model's own answers (C++, onnxruntime "
              << Ort::GetVersionString() << ")\n";
    int failures = 0;
    for (const Check& check : checks) {
        check.report();
        std::printf("  [%s] %s matches %s/reference/\n", check.ok() ? "PASS" : "FAIL",
                    check.name.c_str(), check.name.c_str());
        failures += !check.ok();
    }
    std::cout << (failures ? "FAILED " + std::to_string(failures) + " check(s)\n"
                           : "All checks passed.\n");
    return failures ? 1 : 0;
}

// Run exported NeuroTemplate models from C++: a dynamics model, a control
// policy, or both.
//
// The C++ twin of runtime.py, shipped verbatim in every exported package. It
// reads the same model.json / policy.json and drives the same graphs. Like
// runtime.py it holds no numeric logic: normalisation, window layout, the
// network and the hidden state's seeding all live inside the graphs. This
// file only moves windows of physical values in and out of them.
//
//     model/warm_start.onnx   (y, u, d)          -> (y_next[, state])
//     model/step.onnx         (y, u, d[, state]) -> (y_next[, state])
//     policy/policy.onnx      (y, ymin, ymax, d[, price]) -> u
//
// Every window is a flat, row-major `[steps, n]` array of float32 in physical
// units, oldest row first, with its columns in the order the json lists them.
// Depends on onnxruntime and nlohmann/json only. Header-only, C++17.

#pragma once

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nt {

namespace fs = std::filesystem;
// Ordered: policy.json lists its inputs in the order the graph takes them.
using Json = nlohmann::ordered_json;

// The json layout this runtime reads. Must match runtime.py's FORMAT.
inline constexpr int FORMAT = 1;

inline Json read_json(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    return Json::parse(file);
}

inline Json load_spec(const fs::path& path) {
    Json spec = read_json(path);
    int format = spec.at("format").get<int>();
    if (format != FORMAT) {
        throw std::runtime_error(
            path.string() + " is format " + std::to_string(format) +
            "; this runtime reads format " + std::to_string(FORMAT) +
            ". Use the runtime.hpp shipped alongside it, or re-export.");
    }
    return spec;
}

inline std::vector<std::string> strings(const Json& list) {
    return list.get<std::vector<std::string>>();
}

// One process-wide onnxruntime environment, as onnxruntime requires.
inline Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "neurotemplate");
    return env;
}

// One named graph input: `values` holds `shape` float32s, row-major.
struct Feed {
    const char* name;
    const std::vector<float>* values;
    std::vector<int64_t> shape;
};

// One graph, which must come from the same export as the json beside it.
class Graph {
public:
    Graph(const fs::path& path, const Json& spec)
        : session_(ort_env(), path.c_str(), single_threaded()) {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata meta = session_.GetModelMetadata();
        auto lookup = [&](const char* key) -> std::string {
            auto value = meta.LookupCustomMetadataMapAllocated(key, allocator);
            return value ? value.get() : "(missing)";
        };
        std::string export_id = lookup("export_id");
        std::string opset = lookup("opset");
        std::string expected_id = spec.at("export_id").get<std::string>();
        std::string expected_opset = std::to_string(spec.at("opset").get<int>());
        if (export_id != expected_id || opset != expected_opset) {
            throw std::runtime_error(
                path.filename().string() + " does not belong to the json beside it: graph has "
                "export_id " + export_id + ", opset " + opset + "; json expects export_id " +
                expected_id + ", opset " + expected_opset + ".");
        }
        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            outputs_.push_back(session_.GetOutputNameAllocated(i, allocator).get());
        }
    }

    // Every output's values, flat, in the graph's output order.
    std::vector<std::vector<float>> run(const std::vector<Feed>& feeds) {
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<const char*> input_names;
        std::vector<Ort::Value> tensors;
        for (const Feed& feed : feeds) {
            input_names.push_back(feed.name);
            // onnxruntime only reads inputs; the cast is what its C API asks for.
            tensors.push_back(Ort::Value::CreateTensor<float>(
                memory, const_cast<float*>(feed.values->data()), feed.values->size(),
                feed.shape.data(), feed.shape.size()));
        }
        std::vector<const char*> output_names;
        for (const std::string& name : outputs_) output_names.push_back(name.c_str());

        std::vector<Ort::Value> results = session_.Run(
            Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(),
            output_names.data(), output_names.size());
        std::vector<std::vector<float>> values;
        for (Ort::Value& result : results) {
            const float* data = result.GetTensorData<float>();
            values.emplace_back(data, data + result.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        return values;
    }

private:
    // One thread: these graphs are small enough that thread handoff costs
    // more than it saves at batch 1.
    static Ort::SessionOptions single_threaded() {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        return options;
    }

    Ort::Session session_;
    std::vector<std::string> outputs_;
};

inline void require_size(const std::vector<float>& values, size_t expected, const std::string& what) {
    if (values.size() != expected) {
        throw std::invalid_argument(what + " has " + std::to_string(values.size()) +
                                    " values; expected " + std::to_string(expected) + ".");
    }
}

// ── the dynamics model ──────────────────────────────────────────

// Everything the model needs to take one more step. `outputs` is the latest
// prediction (or, straight after `warmup`, the last measured row), in
// y_columns order. The rest is the model's memory; pass it back to `step`.
struct State {
    std::vector<float> outputs;    // [ny]
    std::vector<float> y_window;   // [history_steps, ny]
    std::vector<float> u_history;  // [history_steps - 1, nu]
    std::vector<float> d_history;  // [history_steps - 1, nd]
    std::vector<float> hidden;     // [state_width]; empty until the first step
    bool warm = false;             // has the first step run?
};

// An open-loop rollout: outputs [steps, ny] and hidden state [steps, state_width].
struct Rollout {
    std::vector<float> outputs;
    std::vector<float> hidden;
};

// An exported dynamics model. `Model("model")` loads a model directory.
//
//   warmup(y, u, d)       the last history_steps rows of every column -> a State
//   step(state, u, d)     one timestep under this step's u and d -> a new State
//   rollout(y, u, d)      warm up on the first history_steps rows, then run
//                         open loop over the rest
class Model {
public:
    explicit Model(const fs::path& dir = "model")
        : spec(load_spec(dir / "model.json")),
          y_columns(strings(spec.at("y_columns"))),
          u_columns(strings(spec.at("u_columns"))),
          d_columns(strings(spec.at("d_columns"))),
          history_steps(spec.at("history_steps").get<int>()),
          timestep_seconds(spec.at("timestep_seconds").get<double>()),
          state_width(spec.at("state").at("width").get<int>()),
          warm_start_(dir / "warm_start.onnx", spec),
          step_(dir / "step.onnx", spec) {}

    const Json spec;
    const std::vector<std::string> y_columns, u_columns, d_columns;
    const int history_steps;
    const double timestep_seconds;
    const int state_width;

    // A State from the last `history_steps` rows of recent measurements, each
    // `[history_steps, n]`. The final u and d rows are not used: they are the
    // inputs to the first `step`.
    State warmup(const std::vector<float>& y, const std::vector<float>& u,
                 const std::vector<float>& d) const {
        const size_t s = history_steps;
        require_size(y, s * ny(), "warmup y");
        require_size(u, s * nu(), "warmup u");
        require_size(d, s * nd(), "warmup d");
        State state;
        state.y_window = y;
        state.u_history.assign(u.begin(), u.end() - nu());
        state.d_history.assign(d.begin(), d.end() - nd());
        state.outputs.assign(y.end() - ny(), y.end());
        return state;
    }

    // Advance one timestep under this step's inputs, `u_now` [nu] and
    // `d_now` [nd]. The State passed in is not modified.
    State step(const State& state, const std::vector<float>& u_now,
               const std::vector<float>& d_now) {
        require_size(u_now, nu(), "step u");
        require_size(d_now, nd(), "step d");
        return advance(state, u_now.data(), d_now.data());
    }

    // Warm up on the first history_steps rows of `y`, `u`, `d` (each
    // `[rows, n]`), then run open loop for `steps` steps (default: to the
    // end). Step k predicts row history_steps + k from the inputs in the row
    // before it; after the warm-up only u and d are read.
    Rollout rollout(const std::vector<float>& y, const std::vector<float>& u,
                    const std::vector<float>& d, int steps = -1) {
        const int s = history_steps;
        const int rows = static_cast<int>(y.size() / ny());
        require_size(u, rows * nu(), "rollout u");
        require_size(d, rows * nd(), "rollout d");
        const int n = steps < 0 ? rows - s : steps;
        if (n < 1 || s + n > rows) {
            throw std::invalid_argument(
                "the frame has " + std::to_string(rows) + " rows; cannot predict " +
                std::to_string(n) + " step(s) after " + std::to_string(s) + " rows of warm-up.");
        }
        State state = warmup({y.begin(), y.begin() + s * ny()}, {u.begin(), u.begin() + s * nu()},
                             {d.begin(), d.begin() + s * nd()});
        Rollout result;
        for (int k = 0; k < n; ++k) {
            const size_t row = s + k - 1;
            state = advance(state, &u[row * nu()], &d[row * nd()]);
            result.outputs.insert(result.outputs.end(), state.outputs.begin(), state.outputs.end());
            result.hidden.insert(result.hidden.end(), state.hidden.begin(), state.hidden.end());
        }
        return result;
    }

    size_t ny() const { return y_columns.size(); }
    size_t nu() const { return u_columns.size(); }
    size_t nd() const { return d_columns.size(); }

private:
    State advance(const State& state, const float* u_now, const float* d_now) {
        const int64_t s = history_steps;
        std::vector<float> u_window = state.u_history;
        u_window.insert(u_window.end(), u_now, u_now + nu());
        std::vector<float> d_window = state.d_history;
        d_window.insert(d_window.end(), d_now, d_now + nd());

        std::vector<Feed> feeds = {
            {"y", &state.y_window, {1, s, static_cast<int64_t>(ny())}},
            {"u", &u_window, {1, s, static_cast<int64_t>(nu())}},
            {"d", &d_window, {1, s, static_cast<int64_t>(nd())}},
        };
        if (state_width && state.warm) feeds.push_back({"state", &state.hidden, {1, state_width}});
        std::vector<std::vector<float>> out = (state.warm ? step_ : warm_start_).run(feeds);

        State next;
        next.outputs = out[0];
        next.y_window.assign(state.y_window.begin() + ny(), state.y_window.end());
        next.y_window.insert(next.y_window.end(), out[0].begin(), out[0].end());
        next.u_history.assign(u_window.begin() + nu(), u_window.end());
        next.d_history.assign(d_window.begin() + nd(), d_window.end());
        if (state_width) next.hidden = out[1];
        next.warm = true;
        return next;
    }

    Graph warm_start_;
    Graph step_;
};

// ── the control policy ──────────────────────────────────────────

// One policy input, as policy.json states it. Rows after `current_row` are
// preview: forecast disturbances, the comfort schedule, the tariff.
struct PolicyInput {
    std::string name;
    std::vector<std::string> columns;
    int steps;
    int current_row;
};

// An exported control policy. `Policy("policy")` loads a policy directory.
// Stateless: one graph, one call per control step.
class Policy {
public:
    explicit Policy(const fs::path& dir = "policy")
        : spec(load_spec(dir / "policy.json")),
          inputs(read_inputs(spec)),
          u_columns(strings(spec.at("u_columns"))),
          graph_(dir / "policy.onnx", spec) {}

    const Json spec;
    const std::vector<PolicyInput> inputs;   // in the order the graph takes them
    const std::vector<std::string> u_columns;

    // The command for one step, [nu]. `windows` maps every input name to its
    // `[steps, n]` window.
    std::vector<float> act(const std::map<std::string, std::vector<float>>& windows) {
        return act_batch(windows, 1);
    }

    // Commands for `batch` windows at once: each input `[batch, steps, n]`
    // -> `[batch, nu]`. For scoring many scenarios at once.
    std::vector<float> act_batch(const std::map<std::string, std::vector<float>>& windows,
                                 int64_t batch) {
        std::vector<Feed> feeds;
        for (const PolicyInput& input : inputs) {
            auto found = windows.find(input.name);
            if (found == windows.end()) {
                throw std::invalid_argument("policy input " + input.name + " not supplied.");
            }
            const int64_t n = static_cast<int64_t>(input.columns.size());
            require_size(found->second, batch * input.steps * n, "policy input " + input.name);
            feeds.push_back({input.name.c_str(), &found->second, {batch, input.steps, n}});
        }
        return graph_.run(feeds)[0];
    }

private:
    static std::vector<PolicyInput> read_inputs(const Json& spec) {
        std::vector<PolicyInput> inputs;
        for (const auto& [name, entry] : spec.at("inputs").items()) {
            inputs.push_back({name, strings(entry.at("columns")), entry.at("steps").get<int>(),
                              entry.at("current_row").get<int>()});
        }
        return inputs;
    }

    Graph graph_;
};

// ── reading the shipped reference ───────────────────────────────

// A CSV whose first column is an index, as every shipped fixture is.
struct Table {
    std::vector<std::string> index;
    std::vector<std::string> columns;   // excluding the index
    std::vector<std::vector<double>> rows;

    // The named columns, `[rows, names.size()]` row-major float32.
    std::vector<float> select(const std::vector<std::string>& names) const {
        std::vector<size_t> positions;
        for (const std::string& name : names) positions.push_back(position(name));
        std::vector<float> values;
        values.reserve(rows.size() * names.size());
        for (const auto& row : rows) {
            for (size_t p : positions) values.push_back(static_cast<float>(row[p]));
        }
        return values;
    }

    size_t position(const std::string& name) const {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (columns[i] == name) return i;
        }
        throw std::out_of_range("the CSV has no column " + name + ".");
    }
};

inline std::vector<std::string> split_csv_line(const std::string& line) {
    if (line.find('"') != std::string::npos) {
        throw std::runtime_error("quoted CSV fields are not supported: " + line);
    }
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

inline Table read_csv(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    Table table;
    std::string line;
    std::getline(file, line);
    std::vector<std::string> header = split_csv_line(line);
    table.columns.assign(header.begin() + 1, header.end());
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error(path.string() + ": a row has " + std::to_string(fields.size()) +
                                     " fields; the header has " + std::to_string(header.size()) + ".");
        }
        table.index.push_back(fields[0]);
        std::vector<double> row;
        for (size_t i = 1; i < fields.size(); ++i) {
            char* end = nullptr;
            row.push_back(std::strtod(fields[i].c_str(), &end));
            if (end == fields[i].c_str() || *end != '\0') {
                throw std::runtime_error(path.string() + ": not a number: " + fields[i]);
            }
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

// The fixture column holding row `t` of input `name`'s `column`. Must match
// runtime.py's `window_columns`.
inline std::vector<std::string> window_columns(const PolicyInput& input) {
    std::vector<std::string> names;
    for (int t = 0; t < input.steps; ++t) {
        for (const std::string& column : input.columns) {
            names.push_back(input.name + "[" + std::to_string(t) + "]." + column);
        }
    }
    return names;
}

// Policy windows stored one per row (policy/reference/inputs.csv) ->
// `{input: [windows, steps, n]}`, ready for `act_batch`.
inline std::map<std::string, std::vector<float>> read_windows(
        const fs::path& path, const std::vector<PolicyInput>& inputs) {
    Table table = read_csv(path);
    std::map<std::string, std::vector<float>> windows;
    for (const PolicyInput& input : inputs) windows[input.name] = table.select(window_columns(input));
    return windows;
}

}  // namespace nt

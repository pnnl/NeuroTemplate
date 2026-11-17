# Export a controller

An export turns a trained controller, or a model, into a small package that
runs without this library: hand it to a colleague, or put it on a device.
Start from a trained controller (`6-first-controller.md`).

1. **Find the run id.** Every training run ends with a line like
   `Finished control run: <run id>`. It's also on the run's page in
   `uv run mlflow ui`.

2. **Export the controller.**

   ```
   nt export policy --run_id <control run id> --out delivery
   ```

   Before writing anything, it runs the package beside the trained
   controller and refuses unless every command matches.

3. **Add the model**, if whoever gets the package should be able to
   forecast with it too. Same folder, so it becomes one package:

   ```
   nt export model --run_id <model run id> --data data/test.csv \
       --out delivery
   ```

   It's checked over every row of `--data`.

4. **Try it.** On any machine with numpy and onnxruntime installed:

   ```
   cd delivery
   python example.py
   python validate.py
   ```

   `example.py` runs it and times one command. `validate.py` checks that
   this machine reproduces the trained results. `README.md` in the folder
   says which columns go in, in what units, and how to call it.

Good to know:

- **Formats.** `--format onnx` (the default) needs numpy and onnxruntime.
  `--format torch` needs torch. `--format c` writes plain C for a
  microcontroller, and needs `onnx2c` installed first (`nt docs export`).
- **The package reads only what the controller was trained to read**: the
  columns in its `policy_window`, with any forecast as far ahead as it was
  trained on. If you can't supply a forecast that long where it will run,
  shorten it and retrain before you export.
- **Re-export into the same folder** to update it. Files you added there
  yourself are left alone.

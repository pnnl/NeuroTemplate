"""A `RunLog` that records what was logged instead of opening an MLflow run."""

from types import SimpleNamespace


class FakeLogger:
    """Every call the pipelines (and Neuromancer's Trainer) make on a logger,
    kept in plain containers for a test to read back."""

    save_weights = False   # Trainer writes no checkpoint of its own

    def __init__(self, run_id: str = "0" * 32, savedir=None):
        self.run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        self.savedir = savedir
        self.metrics, self.dicts, self.texts = {}, {}, {}
        self.artifacts = []

    def log_weights(self, model):
        pass

    def log_metrics(self, output, step=0):
        self.metrics.update(output)

    def log_dict(self, data, name):
        self.dicts[name] = data

    def log_artifacts(self, artifacts=None):
        pass

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((str(path), artifact_path))

    def log_text(self, text, name):
        self.texts[name] = text

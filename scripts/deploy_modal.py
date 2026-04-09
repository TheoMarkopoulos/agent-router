"""Deploy the routing classifier as a Modal serverless endpoint."""

from __future__ import annotations

import modal

app = modal.App("agent-router")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=3.0",
        "scikit-learn>=1.5",
        "pyyaml>=6.0",
        "pydantic>=2.7",
    )
    .copy_local_dir("configs", "/app/configs")
)


@app.cls(image=image, gpu=None, concurrency_limit=10)
class RouterService:
    """Serverless routing classifier backed by sentence-transformer + MLP."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the trained router on container startup."""
        import pickle
        from pathlib import Path

        model_path = Path("/app/models/router.pkl")
        if not model_path.exists():
            self._router = None
            return

        from sentence_transformers import SentenceTransformer

        with open(model_path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        self._clf = payload["clf"]
        self._le = payload["label_encoder"]
        config = payload["config"]
        self._embedder = SentenceTransformer(config["embedding_model"])
        self._router = True

    @modal.method()
    def classify(self, query: str) -> dict[str, str | float]:
        """Classify a query and return the action type with confidence."""
        if self._router is None:
            return {"error": "Model not loaded. Upload models/router.pkl first."}

        import numpy as np

        embedding = self._embedder.encode([query])
        proba = self._clf.predict_proba(embedding)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        action_type = self._le.inverse_transform([best_idx])[0]
        return {"action_type": action_type, "confidence": confidence}

    @modal.method()
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        status = "ok" if self._router is not None else "no_model"
        return {"status": status}


@app.local_entrypoint()
def main() -> None:
    """Smoke-test the deployed service."""
    service = RouterService()
    print("Health:", service.health.remote())
    result = service.classify.remote("What is the capital of France?")
    print("Classify result:", result)

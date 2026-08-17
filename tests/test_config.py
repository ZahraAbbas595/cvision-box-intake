from app.config import ROOT, _resolve_model_path


def test_resolve_model_path_uses_repository_root_for_relative_path() -> None:
    assert _resolve_model_path("models/candidate.pt") == (
        ROOT / "models/candidate.pt"
    ).resolve()


def test_resolve_model_path_preserves_absolute_path() -> None:
    absolute_path = (ROOT / "models/candidate.pt").resolve()

    assert _resolve_model_path(str(absolute_path)) == absolute_path

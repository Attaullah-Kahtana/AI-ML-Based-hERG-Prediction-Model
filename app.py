# Replace the existing load_model() function with this version.

@st.cache_resource
def load_model():
    """Load the first available model from the project directory."""
    configured_path = os.getenv("MODEL_PATH")

    candidates = []

    if configured_path:
        candidates.append(Path(configured_path))

    candidates.extend(
        [
            Path("model.pkl"),
            Path("model.joblib"),
            Path("hERG_model.pkl"),
            Path("hERG_model.joblib"),
            Path("models/model.pkl"),
            Path("models/model.joblib"),
        ]
    )

    # Search recursively if the model has another filename.
    candidates.extend(Path(".").rglob("*.pkl"))
    candidates.extend(Path(".").rglob("*.joblib"))

    checked = set()

    for path in candidates:
        path = path.resolve()

        if path in checked or not path.is_file():
            continue

        checked.add(path)

        try:
            return joblib.load(path), str(path)
        except Exception as error:
            st.warning(f"Could not load `{path.name}`: {error}")

    return None, None

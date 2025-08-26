# Configuration Examples for Calum Worthy AI Assistant

## Dataset Configuration

To change the dataset used by both Streamlit and FastAPI, update the `DATASET_NAME` in your `.streamlit/secrets.toml`:

```toml
# Available datasets:
DATASET_NAME = "calum_v10"    # Current default
# DATASET_NAME = "calum_v11"  # Alternative dataset
# DATASET_NAME = "calum_test" # Test dataset
```

## Prompt File Configuration

To change the persona prompt file, update the `PERSONA_PROMPT_FILE` in your `.streamlit/secrets.toml`:

```toml
# Available prompt files:
PERSONA_PROMPT_FILE = "calum_prompt_v2.1.yaml"  # Current default
# PERSONA_PROMPT_FILE = "calum_prompt_v3.0.yaml" # Future version
# PERSONA_PROMPT_FILE = "calum_experimental.yaml" # Experimental prompts
```

## Environment Variables Alternative

You can also set these as environment variables instead of using secrets.toml:

```bash
export DATASET_NAME="calum_v10"
export PERSONA_PROMPT_FILE="calum_prompt_v2.1.yaml"
```

## FastAPI Configuration

The FastAPI app will automatically:
1. First try to load from `.streamlit/secrets.toml`
2. Fall back to environment variables if secrets.toml is not found
3. Use default values if neither is available

## Streamlit Configuration

The Streamlit app loads directly from `st.secrets` (which reads from `.streamlit/secrets.toml`).

## Configuration Verification

When starting either app, you'll see logs like:
```
✅ Dataset configured: calum_v10
✅ Prompt file configured: calum_prompt_v2.1.yaml
```

This confirms which dataset and prompt file are being used.

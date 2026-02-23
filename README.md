# MLX Audio Wrapper Service

This service provides a FastAPI wrapper around MLX-optimized Qwen3-ASR models. It's built for high-performance audio transcription on Apple Silicon with built-in observability and automatic language detection.

## Setup

1.  **Environment Configuration**:
    Copy the example environment file and adjust settings as needed.
    ```bash
    cp config/service.env.example .env
    ```

2.  **Dependencies**:
    Ensure you have `mlx`, `mlx-audio`, `fastapi`, `uvicorn`, and `prometheus_client` installed.

3.  **Model Path**:
    The service currently uses the model located at:
    `/Volumes/AigoP3500/models/lmstudio/models/mlx-community/Qwen3-ASR-0.6B-bf16`

## Running the Service

Start the server using `uvicorn`:

```bash
python -m uvicorn asr_service.service:app --host 127.0.0.1 --port 8010 --app-dir .
```

## API Usage

### Health Check
Check if the service is up and see the current device status.
```bash
curl http://127.0.0.1:8010/health
```

### Transcription
Transcribe an audio file. The service supports automatic language detection by default.

**Auto Language Detection**:
```bash
curl -X POST http://127.0.0.1:8010/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=auto"
```

**Forced Language**:
```bash
curl -X POST http://127.0.0.1:8010/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=Chinese"
```

### Streaming
Real-time transcription streaming.

**Streaming Request**:
```bash
curl -N -X POST http://127.0.0.1:8010/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "model=/Volumes/AigoP3500/models/lmstudio/models/mlx-community/Qwen3-ASR-0.6B-bf16" \
  -F "response_format=json" \
  -F "stream=true"
```

When `stream=true` and `response_format=json`, the response is SSE (`text/event-stream`) in a vLLM-like format:

```
data: {"id":"transcribe-...","object":"transcription.chunk","created":...,"model":"...","choices":[{"delta":{"content":"..."}}]}

...

data: {"id":"transcribe-...","object":"transcription.chunk","created":...,"model":"...","choices":[{"delta":{"content":""},"finish_reason":"stop","stop_reason":null}]}

data: [DONE]
```

NDJSON mode is also available for simpler parsing by setting `response_format=ndjson`. In that mode you may optionally include the full transcript-so-far in each chunk via `include_accumulated=true`:

```bash
curl -N -X POST http://127.0.0.1:8010/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=ndjson" \
  -F "stream=true" \
  -F "include_accumulated=true"
```

**Output Format**:
Each line is a JSON object:
- `text`: The new text chunk (delta) since the last update.
- `is_final`: Boolean indicating if this is the final result.
- `language`: Detected or forced language.

When `include_accumulated=true`, each line also includes:
- `accumulated`: The full transcript generated so far.

**Note**: Only one streaming request is processed at a time. Concurrent streaming requests return a `429 Too Many Requests` response.

### Metrics
The service exports Prometheus metrics at `/metrics`, including:
- Request counts and durations.
- Active request tracking.
- GPU memory usage (MLX reserved memory).
- Error counts by type.

```bash
curl http://127.0.0.1:8010/metrics
```

## Configuration

Settings are managed via environment variables with the `ASR_` prefix.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ASR_MODEL_PATH` | Path to the MLX model directory | (See `.env.example`) |
| `ASR_HOST` | Server host address | `127.0.0.1` |
| `ASR_PORT` | Server port | `8010` |
| `ASR_MAX_TOKENS` | Maximum tokens for output | `4800` |
| `ASR_CHUNK_DURATION` | VAD chunk size in seconds | `30.0` |
| `ASR_MAX_FILE_SIZE_MB` | Max audio file size | `100` |

## Benchmarking

Use the included stress test script to measure performance across different concurrency levels.

```bash
python benchmarks/asr_stress_bench.py --audio /path/to/test_audio.wav --concurrency 1,2,4
```

The script generates a JSON report at `/tmp/asr_bench_results.json` by default.

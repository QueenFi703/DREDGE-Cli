# DREDGE API Reference

## Python API

### Server Module (`dredge.server`)

#### `create_app() -> Flask`
Creates and configures the Flask application.

**Returns**: Configured Flask app instance

**Example**:
```python
from dredge.server import create_app

app = create_app()
client = app.test_client()
```

#### `run_server(host='0.0.0.0', port=3001, debug=False)`
Run the DREDGE x Dolly server.

**Parameters**:
- `host` (str): Host to bind to (default: '0.0.0.0')
- `port` (int): Port to listen on (default: 3001)
- `debug` (bool): Enable debug mode (default: False)

**Example**:
```python
from dredge.server import run_server

run_server(host='localhost', port=3001, debug=True)
```

#### `_compute_insight_hash(insight_text: str) -> str`
Compute SHA256 hash of insight text with LRU caching.

**Parameters**:
- `insight_text` (str): Text to hash

**Returns**: SHA256 hash as hex string

**Note**: This function is cached with `lru_cache(maxsize=1024)` for performance.

### MCP Server Module (`dredge.mcp_server`)

#### `QuasimotoMCPServer`
Model Context Protocol server for Quasimoto models.

**Methods**:

##### `list_capabilities() -> Dict[str, Any]`
List available models and operations.

**Returns**: Dictionary with server capabilities

**Example**:
```python
from dredge.mcp_server import QuasimotoMCPServer

server = QuasimotoMCPServer()
capabilities = server.list_capabilities()
print(capabilities['capabilities']['models'])
```

##### `load_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
Load a Quasimoto model.

**Parameters**:
- `model_type` (str): Type of model ('quasimoto_1d', 'quasimoto_4d', 'quasimoto_6d', 'quasimoto_ensemble')
- `config` (dict, optional): Configuration (e.g., `{'n_waves': 8}` for ensemble)

**Returns**: Model information including `model_id` and parameter count

**Example**:
```python
result = server.load_model('quasimoto_ensemble', {'n_waves': 16})
model_id = result['model_id']
```

##### `inference(model_id: str, data: Dict[str, Any]) -> Dict[str, Any]`
Run inference on a loaded model.

**Parameters**:
- `model_id` (str): ID of loaded model
- `data` (dict): Input data (e.g., `{'x': [0.5], 't': [0.0]}`)

**Returns**: Inference results with `output` field

##### `get_parameters(model_id: str) -> Dict[str, Any]`
Get parameters of a loaded model.

**Parameters**:
- `model_id` (str): ID of loaded model

**Returns**: Dictionary with parameter values and count

##### `benchmark(model_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
Run a benchmark on a model type.

**Parameters**:
- `model_type` (str): Type of model to benchmark
- `config` (dict, optional): Benchmark configuration (e.g., `{'epochs': 100}`)

**Returns**: Benchmark results with losses and training time

#### `run_mcp_server(host='0.0.0.0', port=3002, debug=False)`
Run the MCP server.

**Parameters**:
- `host` (str): Host to bind to
- `port` (int): Port to listen on
- `debug` (bool): Enable debug mode

### CLI Module (`dredge.cli`)

#### `main(argv=None) -> int`
Main CLI entry point.

**Parameters**:
- `argv` (list, optional): Command-line arguments (defaults to sys.argv)

**Returns**: Exit code (0 for success)

**Commands**:
- `dredge-cli --version` - Show version
- `dredge-cli serve` - Start DREDGE server
- `dredge-cli mcp` - Start MCP server

## REST API Endpoints

### DREDGE Server (Port 3001)

#### `GET /`
API information endpoint.

**Response**:
```json
{
  "name": "DREDGE x Dolly",
  "version": "0.1.4",
  "description": "GPU-CPU Lifter · Save · Files · Print",
  "endpoints": {...}
}
```

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.4"
}
```

#### `POST /lift`
Lift an insight with hash generation.

**Request Body**:
```json
{
  "insight_text": "Your insight text"
}
```

**Response**:
```json
{
  "id": "sha256_hash",
  "text": "Your insight text",
  "lifted": true,
  "message": "Insight processed"
}
```

**Error Response** (400):
```json
{
  "error": "Missing required field: insight_text"
}
```

### MCP Server (Port 3002)

#### `POST /`
Handle MCP protocol requests.

**Request Body**:
```json
{
  "operation": "list_capabilities"
}
```

**Response**: JSON with operation-specific results

## Swift API

### `DREDGECli`
Main Swift CLI entry point.

**Properties**:
- `version` (String): CLI version ("0.1.0")

**Methods**:
- `main()` - Entry point, prints version and message

**Example**:
```swift
// Run from command line
swift run dredge-cli
```

### `SharedStore`
Persistent storage for insights using UserDefaults.

**Methods**:

#### `saveSurfaced(_ text: String)`
Save insight to persistent storage.

**Parameters**:
- `text` (String): Insight text to save

**Example**:
```swift
SharedStore.saveSurfaced("Important insight")
```

#### `loadSurfaced() -> String`
Load saved insight from persistent storage.

**Returns**: Saved insight text, or default message if none saved

**Example**:
```swift
let insight = SharedStore.loadSurfaced()
print(insight)
```

### `QuasimotoWave` (Swift MVP)
Learnable wave function representation (from DREDGE_MVP.swift).

**Properties**:
- `A`: Amplitude parameter
- `k`: Wave number
- `omega`: Temporal frequency
- `v`: Envelope velocity
- `log_sigma`: Log of Gaussian width
- `phi`: Phase offset
- `epsilon`: Modulation strength
- `lmbda`: Modulation frequency

## Quasimoto Models (Python/PyTorch)

### `QuasimotoWave`
1D wave function with 8 parameters.

**Forward Signature**: `forward(x: Tensor, t: Tensor) -> Tensor`

### `QuasimotoWave4D`
4D spatiotemporal wave function with 13 parameters.

**Forward Signature**: `forward(x: Tensor, y: Tensor, z: Tensor, t: Tensor) -> Tensor`

### `QuasimotoWave6D`
6D high-dimensional wave function with 17 parameters.

**Forward Signature**: `forward(x1, x2, x3, x4, x5: Tensor, t: Tensor) -> Tensor`

### `QuasimotoEnsemble`
Ensemble of QuasimotoWave instances.

**Constructor**: `QuasimotoEnsemble(n: int = 16)`
**Forward Signature**: `forward(x: Tensor, t: Tensor) -> Tensor`

## Data Generation Functions

### `generate_data() -> Tuple[Tensor, Tensor, Tensor]`
Generate 1D glitchy chirp signal for benchmarking.

**Returns**: (x, t, y) tensors

### `generate_4d_data(grid_size: int = 16) -> Tuple`
Generate 4D volumetric data.

**Parameters**:
- `grid_size` (int): Grid resolution per dimension

**Returns**: (X, Y, Z, T, signal) tensors

### `generate_6d_data(grid_size: int = 8) -> Tuple`
Generate 6D hyperspace data.

**Parameters**:
- `grid_size` (int): Grid resolution per dimension

**Returns**: (X1, X2, X3, X4, X5, T, signal) tensors

## Training Functions

### `train_model(name, model, x, t, y, epochs=2000, verbose=True, grad_clip=None, use_amp=False)`
Train a Quasimoto model.

**Parameters**:
- `name` (str): Model name for logging
- `model` (nn.Module): Model to train
- `x, t, y` (Tensor): Training data
- `epochs` (int): Number of training epochs
- `verbose` (bool): Print progress
- `grad_clip` (float, optional): Gradient clipping threshold
- `use_amp` (bool): Use automatic mixed precision

**Returns**: (final_loss, losses_list)

## Version Information

Current version: **0.1.4**

Check version:
```bash
dredge-cli --version
```

```python
from dredge import __version__
print(__version__)
```

# Dolly Integration

## Lift Insights with GPU Acceleration

This integration demonstrates the usage of Dolly to efficiently process and transform insights via GPU acceleration (when available).

### Key Components:

1. **`lift_insight()` Function**
   - Accepts an insight as plain text (`str`).
   - Converts text into a tensor using ASCII values.
   - Offloads computations to Dolly to normalize and transform the tensor with GPU support.
   
   #### Steps:
      1. Generate a deterministic `insight_id` using SHA-256 hashing for the text input.
      2. Create a PyTorch tensor by encoding each character of the text (`ascii` values).
      3. Apply Dolly's `lift` for GPU/CPU-based transformation via `heavy_transform` function.
      4. Return a structured dictionary containing:
         - The `insight_id`.
         - Transformed tensor serialized as a list (`vector`).

   ```python
   from dolly import Dolly
   import torch
   import hashlib

   dolly = Dolly()

   def lift_insight(insight_text: str):
       # Deterministic ID (DREDGE-friendly)
       insight_id = hashlib.sha256(insight_text.encode()).hexdigest()

       # Encode insight into tensor (simple + expandable)
       encoded = torch.tensor(
           [ord(c) for c in insight_text],
           dtype=torch.float32
       )

       def heavy_transform(x):
           # Normalize + amplify patterns
           return (x - x.mean()) / (x.std() + 1e-5)

       lifted = dolly.lift(encoded, heavy_transform)

       return {
           "id": insight_id,
           "vector": lifted.tolist()
       }

   # Example Usage:
   insight_text = "Digital memory must be human-reachable."
   result = lift_insight(insight_text)
   print("Insight ID:", result["id"])
   print("Lifted Vector:", result["vector"])
   ```

### Key Benefits:

- **Deterministic IDs:** Generate consistent, unique IDs for insights.
- **GPU Acceleration:** Leverage GPU for heavy computations when available, otherwise fall back to CPU.
- **Extendable:** Replace the transformation logic with more advanced algorithms to suit specific requirements.

## Philosophy

Dolly is a tool. The insights it processes remain under user control. The system emphasizes modular design while ensuring data portability and GPU-empowered transformation.
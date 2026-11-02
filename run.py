from pathlib import Path
import torch
from muscriptor import TranscriptionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model onto your target device
model = TranscriptionModel.load_model("large", device=device)

# Increase batch_size to process multiple 5s chunks concurrently
# Note: prelude_forcing must be False when batch_size > 1
events = model.transcribe_and_postprocess(
    "Taylor Swift - Love Story.wav",
    batch_size=8,              # Process 8 chunks (40 seconds of audio) in parallel
    prelude_forcing=False,     # Disable sequential autoregressive conditioning across seams
)

# Save the resulting MIDI bytes
Path("Taylor Swift - Love Story.mid").write_bytes(events[0])
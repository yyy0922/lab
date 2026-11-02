# SAS-Droid implementation

Use the approved v1.0 design, in a separate directory; retain LTA sources unchanged.

1. Freeze the original opcode map and train-only GA selector with source hashes. Extract all labeled APKs under ../apk/benign and ../apk/malradar; inventory ../apk/new without inventing labels. Cache complete method sequences and source identities.
2. Test representation states, valid-prefix reversal/padding invariance, method/APK aggregation and gradients through group offsets. Implement portable PyTorch Mamba-1 selective scan (explicit recurrence, not the official CUDA kernel).
3. Implement C0–C6, A/B/C/D, delta/BC ablations and F0/F1/F2. Keep all methods and positions; exact duplicate-method compression retains multiplicities. Use staged method microbatches and recomputation for bounded training memory.
4. Group identical APK feature vectors before splitting, then freeze train/validation/test. GA sees training only. Persist checkpoints, validation selection, resource timing and predictions. Test evaluation is an explicit final command, not part of tuning.
5. Run unit tests and real-APK extraction/training. Report completed evidence separately from remaining research experiments; no reproduction or superiority claim without measured results.

Decisions: CPU-only host needs a transparent slow scan implementation. A CPU smoke configuration is allowed but must be explicitly named, never substituted for the 64-dimensional research configuration. No silent truncation, method sampling, or labels for new/.

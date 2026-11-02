# Sources and attribution

This research project is based on the user-provided LTAChecker/LTA-main source.
The original opcode map and reproduction runner are retained in reference/;
reference/provenance.json records their source SHA-256 hashes. The GA/SVM
selector is reused from that runner. This project does not claim ownership
of those third-party portions or alter their applicable rights.

Reference paper: Liu et al., LTAChecker (2024),
https://doi.org/10.1109/JIOT.2024.3394555

tools/apktool.jar is the APKTool binary supplied with the original source,
included for optional static extraction. Upstream project and licensing:
https://github.com/iBotPeaches/Apktool

The model is a PyTorch implementation of the Mamba-1 selective recurrence,
not the official fused Mamba CUDA package. GPU execution does not imply use
of the official kernels.

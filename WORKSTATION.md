# Windows / CUDA 13 setup

`.venv` is the Python 3.12 inference/test environment, using PyTorch 2.14.0+cu130 and ONNX Runtime GPU 1.30.0. `.venv-export` isolates the upstream export dependencies, including NumPy 1.26.4 and OpenCV 4.11. Install Git, uv and the Visual Studio C++ build tools before running setup.

```powershell
.\setup-windows.ps1
.\run-native.ps1 --help
```

The setup script uses this repository's `python/fetch_sam3_repo.py` to fetch the
public `facebookresearch/sam3` source at the revision pinned in
`python/sam3_revision.py`. By default the checkout is `../sam3`. To reuse another
checkout, pass `-Sam3Repo C:/path/to/sam3`; use the same path with the exporter's
`--sam3-repo` option. Setup synchronizes that checkout to the pinned revision.
No private repository is required and setup does not download model weights.

Run `export/onnx_export.py` using `.venv-export/Scripts/python.exe`; its Windows
compatibility shim handles optional unavailable Triton modules.

Python is installed under `%USERPROFILE%\.local\share\uv\python`, outside
AppData's packaged-app redirection. Set `UV_PYTHON_INSTALL_DIR` to that location
before creating new environments; `setup-windows.ps1` uses it by default.

For approved access to Meta's gated checkpoint, run `./login-huggingface.cmd`
in your own terminal. It runs the Hugging Face login using the base interpreter
recorded in `.venv-export/pyvenv.cfg`, avoiding the venv launcher. Complete the
browser login or enter a read token only at its private prompt. Checkpoint access
requires your own approved Hugging Face account. Never store a token in source
files, command examples or Git.

To rebuild C++, use an x64 Visual Studio Developer PowerShell. Set `OpenCV_DIR`
to the directory containing `OpenCVConfig.cmake` and `ONNXRUNTIME_ROOT` to the
ONNX Runtime SDK directory, or pass the CMake cache paths explicitly:

```powershell
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR=C:/path/to/opencv/build `
  -DONNXRUNTIME_DIR=C:/path/to/onnxruntime
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure
```

The launcher supplies PyTorch's CUDA/cuDNN DLL directory for this process. Fetch
or export the model assets separately as described in [README.md](README.md).
Keep weights, authentication files and local input/output data outside Git.

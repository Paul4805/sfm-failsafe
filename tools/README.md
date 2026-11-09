# tools/

Portable external binaries live here. They are ignored by git except this file.

For this checkout, COLMAP 3.9.1 Windows CUDA was placed at:

```
tools/colmap391cuda/COLMAP-3.9.1-windows-cuda/bin/colmap.exe
```

The runner resolves the sibling `lib/` directory automatically, so
`configs/courtyard.yaml` points at the `.exe` rather than `COLMAP.bat`.

Why 3.9.1? Windows Smart App Control blocked the newer unsigned 4.2.1 release
and pycolmap DLL with error `0xC0E90002`. The older 3.9.1 release was allowed
and ran successfully. This is an environment-specific workaround; do not treat
it as a COLMAP version requirement. On an unrestricted machine, install any
supported COLMAP release and set `colmap.executable`.

Verify:

```bash
tools/colmap391cuda/COLMAP-3.9.1-windows-cuda/bin/colmap.exe -h
```

Download sources:

- https://github.com/colmap/colmap/releases
- https://www.eth3d.net/datasets

import importlib.util
import os
import sys

# UMI ships a package also called `diffusion_policy`, distinct from RMB's third_party/diffusion_policy.
# Insert at the front so UMI's wins, and import this module before any `diffusion_policy` import.
# Only one policy module is loaded per process (see bin/Train.py), so the two never coexist.
UMI_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../third_party/universal_manipulation_interface",
    )
)

if os.path.isdir(UMI_PATH):
    if UMI_PATH not in sys.path:
        sys.path.insert(0, UMI_PATH)
elif importlib.util.find_spec("diffusion_policy") is None:
    raise ImportError(
        f"[UmiDp] UMI source not found at {UMI_PATH}, and 'diffusion_policy' is not importable.\n"
        "Add it as a submodule:\n"
        "  git submodule add https://github.com/real-stanford/universal_manipulation_interface.git "
        "third_party/universal_manipulation_interface"
    )

import os
import shutil

src = r"F:\AppsCrucial\ComfyUI_phoenix2\ComfyUI\models\loras\zimage\b0bb13_g00ds_zimage"
dst = r"F:\AppsCrucial\ComfyUI_phoenix2\ComfyUI\models\loras\zimage\b0bb13_g00ds_zimage_OLD"

if os.path.exists(src):
    try:
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")
    except Exception as e:
        print(f"Error renaming: {e}")
else:
    print(f"Source path does not exist: {src}")
    # List parent dir to debug
    parent = os.path.dirname(src)
    if os.path.exists(parent):
        print(f"Contents of {parent}:")
        for item in os.listdir(parent):
            print(f" - {item}")
    else:
        print(f"Parent path does not exist: {parent}")

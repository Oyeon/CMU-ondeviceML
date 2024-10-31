from ultralytics import YOLO
import shutil

# Helper function to export and move model
def export_and_save(model, format_type, save_path, **export_kwargs):
    try:
        exported_model_path = model.export(format=format_type, **export_kwargs)
        if exported_model_path:
            shutil.move(exported_model_path, save_path)
            print(f"Model successfully saved to {save_path}")
        else:
            print("Error: Exported model path not found.")
    except Exception as e:
        print(f"Export failed for {save_path}: {e}")

# Load model
model = YOLO('best.pt')

# int8 export
export_and_save(model, format_type="engine", save_path='int8_static.engine', int8=True, imgsz=640, device="cuda:0")

# fp16 export
export_and_save(model, format_type="engine", save_path='fp16_static.engine', half=True, imgsz=640, device="cuda:0")

# fp32 export
export_and_save(model, format_type="engine", save_path='fp32_default.engine', imgsz=640, device="cuda:0")


# from ultralytics import YOLO
# import shutil

# # int8
# save_path = 'int8_static.engine'  # Adjust the path as needed
# model = YOLO('best.pt')
# exported_model_path = model.export(format="engine", int8=True, imgsz=640, device=0)

# # Locate and move the exported model to the target location
# if exported_model_path:
#     shutil.move(exported_model_path, save_path)
#     print(f"Model successfully saved to {save_path}")
# else:
#     print("Error: Exported model path not found.")

# # fp 16
# save_path = 'fp16_static.engine'  # Adjust the path as needed
# model = YOLO('best.pt')
# model.export(format="engine", half=True, imgsz=640, device=0)

# # Locate and move the exported model to the target location
# if exported_model_path:
#     shutil.move(exported_model_path, save_path)
#     print(f"Model successfully saved to {save_path}")
# else:
#     print("Error: Exported model path not found.")


# # fp 32
# save_path = 'fp32_default.engine'  # Adjust the path as needed
# model = YOLO('best.pt')
# model.export(format="engine", imgsz=640, device=0)

# # Locate and move the exported model to the target location
# if exported_model_path:
#     shutil.move(exported_model_path, save_path)
#     print(f"Model successfully saved to {save_path}")
# else:
#     print("Error: Exported model path not found.")


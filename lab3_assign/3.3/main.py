# from config import CONFIG
# from train import main_training_pipeline
# import os
# import json

# def main():
#     """
#     Entry point for training and evaluating YOLOv8 models.
#     """
#     # Ensure the configuration enables YOLOv8 training
#     if not CONFIG.get('enable_yolo', False):
#         print("YOLOv8 training is disabled in the configuration.")
#         return
    
#     print("Starting YOLOv8 training and evaluation pipeline...")
    
#     # Start the training and evaluation pipeline
#     main_training_pipeline(CONFIG)
    
#     # Optionally, load and display results
#     home_dir = os.getcwd()
#     results_path = os.path.join(home_dir, 'results.json')
    
#     if os.path.exists(results_path):
#         with open(results_path, 'r') as f:
#             results = json.load(f)
#         print("Training and Evaluation Results:")
#         print(json.dumps(results, indent=4))
#     else:
#         print(f"No results found at {results_path}")

# if __name__ == "__main__":
#     main()
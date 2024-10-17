CONFIG = {
    'dataset_train_path': '../../datasets/lab1_dataset/mnist_train.csv',
    'dataset_test_path': '../../datasets/lab1_dataset/mnist_test.csv',
    'input_dims': 28 * 28,
    'num_hidden_layers': 2,
    'hidden_layer_width': 1024,
    'hidden_feature_dims': 1024,
    'output_classes': 10,
    'train_batch_size': 64,
    'test_batch_list': [1, 64],
    'test_batch_size': 1,
    'learning_rate': 0.001,
    'epochs': 2,
    'inference_device': 'cpu', #'cpu'
    'transform_type': 'crop_20', #'resize_14', 'resize_20', 'crop_20', 'no_transform'
    'quantization': 'qat', #'dynamic', 'static', 'qat','none'
    'qat_epochs': 2, 
    'model_save': False, 
    # 'activation_function': 'relu', #'relu', 'tanh', 'gelu'
    # 'dropout_prob': 0.8, #(0.5, 0.8, 1)
    # 'weight_init': 'torch_default', # 'torch_default', 'he', 'xavier', 'random', 'zero_one'
    # 'weight_decay': 0.0, # 0.0, 0.01, 0.001
}
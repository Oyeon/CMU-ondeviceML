CONFIG = {
    'dataset_train_path': '../datasets/lab1_dataset/mnist_train.csv',
    'dataset_test_path': '../datasets/lab1_dataset/mnist_test.csv',
    'input_dims': 28 * 28,
    'hidden_feature_dims': 1024,
    'output_classes': 10,
    'train_batch_size': 64,
    'test_batch_size': 10,
    'learning_rate': 0.001,
    'epochs': 2,
    'device': 'gpu', #'cpu'
    'transform_type': 'resize_20' #'resize_14', 'resize_20', 'crop_20', 'no_transform'
}
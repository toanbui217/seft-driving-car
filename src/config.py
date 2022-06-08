import os

# direction
parent_dir = os.path.abspath("..")

data_img_dir = os.path.join(parent_dir, 'data', "IMG")
data_log_dir = os.path.join(parent_dir, 'data', "driving_log.csv")

log_dir = os.path.join(parent_dir, 'output', "logs")
model_dir = os.path.join(parent_dir, 'output', "models")

"""
Các thông số data:
    ảnh camera chính,
    ảnh camera trái,
    ảnh camera phải,
    góc lái,
    độ giảm tốc độ,
    phanh,
    tốc độ,
"""
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
columns_need = ['center', 'left', 'right', 'steering']

# Input shape
img_width = 200
img_height = 66
img_channels = 3
input_shape = (img_height, img_width, img_channels)

# Model config
model_params = {
    'epochs': 10,
    'learning_rate': 1.0e-4,
    'drop_rate': 0.5,
    'epochs_drop': 20,
    'verbose': 1,
    'batch_size': 32,
    'steps_per_epoch': 1000,
    'validation_steps': 5
}

unique_name = str(model_params["learning_rate"]) + "-lr" \
              + str(model_params["epochs"]) + "-e" \
              + str(model_params["steps_per_epoch"]) + "-spe" \
              + str(model_params["epochs_drop"]) + "-ed" + "--v2"

ratio_split = 0.8

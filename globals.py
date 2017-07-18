"""
constants for the data set.
ModelNet40 for example
"""
NUM_CLASSES = 40
NUM_VIEWS = 12
TRAIN_LOL = './data/view/train_lists.txt'
VAL_LOL = './data/view/val_lists.txt'
TEST_LOL = './data/view/test_lists.txt'


"""
constants for both training and testing
"""
BATCH_SIZE = 16

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * BATCH_SIZE


"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001

# sample how many shapes for validation
# this affects the validation time
VAL_SAMPLE_SIZE = 256

# do a validation every VAL_PERIOD iterations
VAL_PERIOD = 100

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.
SAVE_PERIOD = 1000


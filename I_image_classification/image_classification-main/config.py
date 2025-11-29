import os

# use this setup to store data relative to the working directory:
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data/')

# use this setup when working on bwJupyter using existing data download
#data_dir = "~/work/__shared/bonus_assignment_1_image_classification/data"

if not os.path.isdir(os.path.join(current_dir, 'weights')):
    os.mkdir(os.path.join(current_dir, 'weights'))
model_checkpoint_file = os.path.join(current_dir, 'weights/checkpoint.pt')
model_best_checkpoint = os.path.join(current_dir, 'weights/best.pt')
# Bonus Assignment: Image Classification

In this assignment, you will be training your own neural network prediction model for the task of aerial image classification.
Parts of the training script, and especially the access and organization of the dataset is already implemented,
but small parts, and especially the model and hyperparameter choices need to be implemented/completed.

*The assignment was created and tested using Python 3.12.*

## Aerial Image Classification - Task and Dataset

In the task of aerial image classification, the goal is to predict one of 10 land use types based on 64x64 RGB images.
The 10 classes are

1. AnnualCrop
2. Forest
3. HerbaceousVegatation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

The corresponding dataset consists of 27,000 annotated images is available for download using PyTorch.
The total size is approximately 90MB, it will be downloaded when you first run the python file `data_loader.py`,
or on first execution of the training script.

The data will be downloaded and extracted to the project directory's subfolder `./data`.
After downloading, you can browse the directory `./data/EuroSAT` to manually inspect the dataset images, if you wish.

#### Note for Users Working on bwJupyter

If you are working on bwJupyter, first copy the python files (*.py) and the requirements.txt from the `__shared`-directory to your personal workspace.
Best create a new directory for the assignment and copy the files there.

You do not need to download the dataset yourself as the data is already downloaded to the shared storage at '~/work/__shared/bonus_assignment_1_image_classification/data'.
Make sure the data path in `config.py` is set accordingly.


## This Repository

In this repository, the following files already exist:

* `config.py`: Contains basic configuration of paths which is required by multiple code files
* `data_loader.py`: Code for data download and organisation of the data into PyTorch datasets (includes training-validation-test split)
* `model.py`: This is the skeleton of the neural network prediction model. Implementation pending.
* `train.py`: The complete training script. Some parts need to be completed, hyperparameters need to be set.
* `evaluate.py`: Script to evaluate the fully trained model on the test data. Displays confusion matrix.

## Assignment Details

To pass the assignment, you need to do the following:

* **Complete the training script** in `train.py`. Sections where you need to fill missing code are commented by `# TO DO:...`
* **Implement a neural network model** in `model.py`. Similar to the tutorial's exercise, initialize the model's layers in the class' `__init__`-method and implement the forward pass in `forward()`.
* **Find suitable training hyperparameters** in `train.py`. At the top of the script, right after the import statements, some important training hyperparameters are set. Find suitable settings which allow for a successful model training.
* **Achieve an accuracy** of 0.5 or higher on the test data. (The existing dataset splits may not be adjusted.)

Once you have a first implementation, you can run the training procedure using the training script `train.py`.
The training script will
* Download the dataset (if not already available) to the subfolder `./data`
* Train the model and write training logs to `./runs`
* Store the model weights to `./weights`

After training is completed, the evaluation procedure `evaluate.py` can be executed.
The script will load your trained model and perform an evaluation on the test dataset.
The resulting accuracy is printed to your console and a confusion matrix is displayed.
After you have finished your work, this accuracy value should exceed `0.5`.

*Hints:*
* **Python environment setup**: The file `requirements.txt` contains a list of all packages required to complete the assignment. Before starting your work on the assignment, you can run the command `pip install -r requirements.txt` in your console (make sure to choose the assignment's base directory first so that the requirements-file can be found).
* **Monitor your progress and results using tensorboard**: Tensorboard is a very useful tool to easily access training progress visualizations and compare multiple training runs. The training-script in this assignments writes logfiles using PyTorch's `SummaryWriter`. These files can be read and visualized using tensorboard. After all required packages are installed, you can run tensorboard from your command line using `tensorboard --logdir .`. Subsequently, you can open tensorboard's UI in your browser by navigating to `localhost:6006`.
* **Achieving the required accuracy:** The accuracy can be achieved by editing only the code parts marked as `# TO DO`. You do not need to do additional changes to the training code. In our tests, we were able to train a suitable model on a CPU in approximately 5 minutes of training time. Off course, you may not modify the evaluation procedure.

#### Note for Users Working on bwJupyter

Monitoring your training progress on bwJupyter using tensorboard is currently not supported.

## Submission

You can submit your assignment via ILIAS.

At the top of the file `train.py`, please fill your name, identifier, 
and - if applicable - the name of your collaborating partners in the designated section before submission.

Please create a *.zip (or *.tar.gz) archive of your modified source code files (i.e. `train.py` and `model.py`) and 
the training logs of your final model training in the `runs`-folder.
You are not required to include your trained model (the `weights`-folder) in your submission.

If working on bwJupyter, the command
```bash
tar -cvzf submission.tar.gz train.py model.py runs/
```
might be used to create the archive.
Subsequently, download the created `submission.tar.gz`-file to your local computer 
and upload it to ILIAS.
and upload it to ILIAS.

If you require assistance with the coding, the submission, or other technical issues, please contact us. 
We are happy to help!
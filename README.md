# Automated_Material_Recognition

## Setup Guide

* A .env file should be configured in the project root directory i.e. within Automated_Material_Recongition that includes the following defined paths
    * PROJECT_DIR -> An absolute path to the project root directory e.g. C:/Users/jackb/source/repos/Automated_Material_Recognition
    * DATA_DIR -> A relative path from the project root directory to a directory containing the images (zip file) e.g. ./data
    * LOG_DIR -> A relative path from the project root directory to a whereever you'd like the output from the logger to be stored e.g. ./logs

* You must place the zip file containing the train and test images inside your DATA_DIR

* Requirements can be found in a requirements.txt file within the project root directory, all versions of dependancies are compatible with pip version 25.0.1

* Trained models and checkpoints will be saved to ./models, you'll find an archived model under the archive directory.

* Exploratory work with Ollama can be found at the bottom of main.ipynb, this was later de-prioritised, justification can be found in the file.
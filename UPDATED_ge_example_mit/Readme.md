#Great Expectations

##Introduction 

This project contains expectations created using a mock training dataset. These expectations are then run on another mock validation dataset to see how many pass. The more expectations that pass, the more similar the training and validation datasets are.

Note this project provides a simple example of expectations to give an idea of what the great expectations package is capable of, for full understanding of how expectations can be adapted for your needs please visit the comprehensive documentation for great expectations https://legacy.docs.greatexpectations.io/en/latest/guides/how_to_guides.html. 

There is a V2 and V3 of great expectations, in this project we have used V3. This project was created using the steps in the getting started guide https://legacy.docs.greatexpectations.io/en/latest/guides/tutorials/getting_started_v3_api.html. 

In this project data/example_data.csv was used to build the expectations and example_data_for_validation.csv was used to validate the data against the checkpoint. 

To run the expectations in this project follow these steps in the terminal:

1. Install great expectations using 'pip install great_expectations'
2. To run the validation enter the command 'great_expectations --v3-api checkpoint script example_checkpoint
'. This will create a python file in an 'uncommitted' folder. Run this python file using the command 'python great_expectations/uncommitted/run_example_checkpoint.py'
   
3. You will now see a 'data_docs' folder has been created. Go to validations/example_suite. This will contain a folder with a html page where you can view the outcome of the expectations. 
   


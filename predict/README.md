# Create predictions from test images.

For generating test results run the following from the repositorys root directory:  
```bash
python predict/generate_test_csv.py --model <model_path>
```

The **output** file will be in the output directory. The **sorted_test_data_prediction.csv** is ready to upload to the submission interface.

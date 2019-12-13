# Extract rooftops from Caribbean challenge TIFF 

For extract images run the following from the projects root directory:  
```bash
python extract_images [-t, --test]
```

If the **-t** or the **--test** flag is passed then the module will extract the test images. Else it will always extract the training rooftops.

The Competitions **stac** folder is needed unchanged.

The output images will in **stac/datasets/training_data** or **stac/datasets/test_data** whether we chose the extract testing data option.

# Graph dataset hunt project: Text-Image Web News

## Data source
Download at 
https://sites.google.com/site/qianmingjie/home/datasets/cnn-and-fox-news

## Data processing
Example;
```
opt = {
    "data_dir" = '/Users/mob/Documents/Datasets/CNN',
    "dataset_str" = 'cnn',
    "similarity_threshold" = 0.1,
    }
```
in `<data_processing.py>` 
* `graph_construction(opt)`: to construct graph and return giant component
* `graph_visualization(opt)`: to construct graph and visualize giant component
* `gcn_input(opt)`: to construct graph and prepeare input data for gcn (data split and formatting: returns standard inputs of gcn, returned variables are same in name compatible in fomat)

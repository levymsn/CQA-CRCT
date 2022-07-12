A short explanation about the config file:
<pre>
{
    "name": the name of the config
    "dataset": the name of the dataset
    "categories": A magic number for visual embeddings categories, to include colors and shapes
    "max_vis_features": number of visual features per input (~ nubmer of visual objects appear in the figure)
    "max_seq_len": number of text tokens per input (~questions length)
    "binary_answers": A flag for a binary answers dataset
    "main_folder": the FULL PATH to the CRCT folder (../)
    "model_config": ViLBERT parameters,
    "save_path": path to folder where to save experiments,
    "figure_feat_path": a path to the figure features that was extracted from the detector.,
    "qa_parent_dir": a path to the Q&As files folder
    "tensorboard": a path to where to save tensorboard informations,
    "checkpoints_dir": pretrained checkpoints to start from
    "dataset_files_divisions": this is a dictionary that contains the number of images appear in each figure features file.
    "splits": the name of the dataset's splits
}
</pre>
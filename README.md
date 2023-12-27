# DAMAF:Dual Attention Network with Multi-level Adaptive Fusion for Medical Image Segmentation

## How to use
### Training and Testing

1. Download the Synapse dataset.

2. Run the following code to install the Requirements.

   `pip install -r requirements.txt`

3. Run the below code to train the DAEFormer on the synapse dataset.

   ```bash
   python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 16 --eval_interval 20 --max_epochs 400
   ```

   **--root_path**     [Train data path]

   **--test_path**     [Test data path]

   **--eval_interval** [Evaluation epoch]

 4. Run the below code to test the DAEFormer on the synapse dataset.

    ```bash
    python test.py --volume_path ./data/Synapse/ --model './model_out/best_model.pth'
    ```

    **--volume_path**   [Root dir of the test data]
        
    **--model**    [Your learned weights]

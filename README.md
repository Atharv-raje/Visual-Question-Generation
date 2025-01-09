# Visual Question Generation in TensorFlow

This repository implements a Visual Question Generation (VQG) system that generates natural language questions based on image content. Inspired by the paper *[Generating Natural Questions About an Image](https://arxiv.org/abs/1603.06059)*, the model uses a combination of image features extracted from a pre-trained VGG19 network and an LSTM-based question generator.

---

## Features
- Extract image features using a pre-trained VGG19 model.
- Generate natural language questions using LSTM.
- Trainable on the VQA dataset with plans to integrate the VQG dataset.

---

## Requirements
- **Python 2.7**
- TensorFlow (v0.10)
- Keras
- NumPy
- OpenCV
- Skimage
- Pre-trained VGG19 weights ([Download here](https://github.com/machrisaa/tensorflow-vgg))
- VQA dataset ([Download here](http://www.visualqa.org))

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/visual-question-generation.git
   cd visual-question-generation
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the VQA dataset and preprocess it using the [VQA LSTM CNN repo](https://github.com/VT-vision-lab/VQA_LSTM_CNN). Place the processed files (`data_prepro.h5`, `data_prepro.json`, `data_img.h5`) in the root directory.
4. Download the pre-trained VGG19 weights and place them in the root directory.

---

## Usage

### Train the Model
To train the VQG model:
```bash
python main.py --is_train=True --model_path=[where_to_save]
```

### Generate Questions for a Single Image
To run a demo with a single image:
```bash
python main.py --is_train=False --test_image_path=[path_to_image] --test_model_path=[path_to_model]
```

### Configuration
Modify the following flags in `main.py` for custom settings:
- `input_img_h5`: Path to the H5 file containing image features.
- `input_ques_h5`: Path to the H5 file containing preprocessed questions.
- `input_json`: Path to the JSON file containing vocabulary and metadata.
- `batch_size`, `dim_embed`, `dim_hidden`, `learning_rate`, and other hyperparameters.

---

## Files
- **`vgg19.py`**: Defines the VGG19 network for extracting image features.
- **`data_loader.py`**: Preprocesses dataset files and normalizes image features.
- **`question_generator.py`**: Implements the LSTM-based question generation model.
- **`main.py`**: Main script to train or test the model.

---


## Known Issues
- The VQA dataset is not ideal for natural question generation since its questions are designed for visual question answering challenges.
- Requires upgrading to TensorFlow 2.x for modern compatibility.

---

## Future Work
- Integrate the VQG dataset for more natural question generation.
- Upgrade codebase to Python 3 and TensorFlow 2.x.
- Implement a Transformer-based question generator.

---

## References
- [Generating Natural Questions About an Image](https://arxiv.org/abs/1603.06059)
- [VQA LSTM CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN)
- [TensorFlow VGG](https://github.com/machrisaa/tensorflow-vgg)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

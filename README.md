
# Pocket Racer: An Open-Source, Head-to-Head Autonomous Racing Platform for AI Education
![final4 (1)-1](https://github.com/PocketRacers/PocketRacerRepo/assets/119138249/5c117452-3276-4f5c-b0bc-597bf2cf3c60)
## Abstract
Head-to-head autonomous racing can engage undergraduate students in artificial intelligence (AI) and robotics education through the excitement of maneuvering sharp corners and overtaking opponents. In particular, the construction and programming of miniature mobile racing robots can facilitate head-to-head racing competitions for use in indoor classrooms and hallways. Unfortunately, existing racing platforms remain inaccessible for overtaking maneuvers in such confined settings because they are physically large or expensive due to the computational cost of enabling such high performance in autonomy. In an attempt to address these issues, we present Pocket Racer, an open-source, pocket-sized racing robot capable of head-to-head racing within a typical hallway or classroom. Inspired by how humans visually perceive the sequential progression of the road while driving, we propose an efficient yet effective approach to end-to-end autonomous driving. We demonstrate head-to-head autonomous racing with our Pocket Racer platforms, enabling high speed overtaking within hallways upwards of 15 km/h. Designed to be easily assembled with off the shelf components and a low-cost edge device (Raspberry Pi Zero 2 W), our Pocket Racer platform is made fully accessible through an open source website detailing build instructions. By making head-to-head autonomous racing accessible for undergraduate students, our work hopes to further artificial intelligence and robotics education.

## Project Description
Pocket Racer is a platform designed for simulating and training multi-agent autonomous racing models. Leveraging advanced transformer architectures, this platform enables the study and improvement of autonomous racing strategies in robotics education.

## Installation

To set up the Pocket Racer platform, please follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourgithub/pocket-racer.git

# Navigate to the project directory
cd pocket-racer

# Install required dependencies
pip install -r requirements.txt
```

## Usage

## Data Processing

To use the `data_processor.py` script for preparing and processing your datasets, run the following command:

```bash
python data_processor.py --n_stacked <number_of_stacked_images> --img_path <path_to_images> --csv_path <path_to_csv> --w <width> --h <height> --d <depth> --concatenate <concatenate_flag> --prediction_mode <mode>
```

Modify the parameters according to your dataset's requirements:

- **n_stacked**: Number of images to stack for input data.
- **img_path**: Path to the directory containing image files.
- **csv_path**: Path to the CSV file containing metadata or labels.
- **w, h, d**: Width, height, and depth of the processed images.
- **concatenate**: Boolean flag to determine whether to concatenate data.
- **prediction_mode**: Specifies the mode of prediction, e.g., 'linear' or 'categorical'.
## Training
To train the model using the provided scripts, you can run:

```bash
python train_script.py --epochs 50 --batch_size 1000
```

Modify the parameters as needed for your specific training setup.

## Features

- **Model Training**: Train vision transformers tailored for steering prediction.
- **Performance Metrics**: Evaluate model performance with metrics such as MAE, RMSE, and R².
- **Customizable Training Options**: Configure training epochs, batch size, and image preprocessing through command-line arguments.

## Contributing

Contributions to Pocket Racer are welcome! Please consult the `CONTRIBUTING.md` file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Authors and Acknowledgment

- Eun Sang Cha
- Junu Choe
- Kittimate Chulajata
- Sean Wu
- Dennis Hong*
- Daekyum Kim*

We thank all contributors who have helped in developing and refining this platform.
## Contact Information

For any queries or support, please contact [jesse.esc.cha@gmail.com].
## Citation
If you use this software in your research, please cite it as follows:

```bibtex
@article{PocketRacer2024,
  title={Pocket Racer: An Open-Source, Head-to-Head Autonomous Racing Platform for AI Education},
  author={Cha, Eun Sang and Choe, Junu and Chulajata, Kittimate and Wu, Sean and Hong, Dennis and Kim, Daekyum},
  journal={Advanced Intelligent Systems},
  year={2024},
  note={Submitted}
}


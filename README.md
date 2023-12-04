# StarGAN-TA(StarGAN with Transfer Learning and changed Activation function)

* Data Composition

  We compose data for 6 facial expressions. If want to add more facial expressions like (annoyed, scared), add data folder in the same location(--rafd_image_dir). Empirically we find out that more than 500 pictures are needed for good results.

  - Own StarGAN Folder

    - Data

      - train

        - happy

        - sad

        - surprised

        - neutral

        - fearful

        - Angry

          

* Training

  ``` cmd
  python3 main.py --mode train --data RaFD --image_size 128 --   model_save_dir='stargan_new_6_leaky/models' --result_dir='stargan_new_6_leaky/result' --rafd_image_dir='stargan_new_6_leaky/data/train' --sample_dir='stargan_new_6_leaky/samples' --sample_label_dir='stargan_new_6_leaky/samples'
  ```

  

* 3 modes for this model(test data location is fixed so we notify test data location in code.)

  1. Change face attribution(ex. Smile, sad, ..) for one person(cropped image) - person

     ```cmd
     python3 main.py --mode test --data RaFD --image_size 128 --model_save_dir='stargan_new_6_leaky/models' --result_dir='stargan_new_6_leaky/results' -- sample_dir='stargan_new_6_leaky/samples' --sample_label_dir='stargan_new_6_leaky/data/train' --test_mode person --test_iters 330000
     ```

     

  2. Change face attribution for multi person(No need to crop) - group

     ```cmd
     python3 main.py --mode test --data RaFD --image_size 128 --model_save_dir='stargan_new_6_leaky/models' --result_dir='stargan_new_6_leaky/results' -- sample_dir='stargan_new_6_leaky/samples' --sample_label_dir='stargan_new_6_leaky/data/train' --test_mode group --test_iters 330000
     ```

     

  3. Change face attribution for one person(No need to crop) - origin_person

     ```cmd
     python3 main.py --mode test --data RaFD --image_size 128 --model_save_dir='stargan_new_6_leaky/models' --result_dir='stargan_new_6_leaky/results' -- sample_dir='stargan_new_6_leaky/samples' --sample_label_dir='stargan_new_6_leaky/data/train' --test_mode origin_person --test_iters 330000
     ```

  Inferenced results are saved in args.result_dir -> user can set the directory

* Results

  
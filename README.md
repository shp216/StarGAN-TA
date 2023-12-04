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
  python3 main.py --mode test --data RaFD --image_size 128 --   model_save_dir='stargan_new_6_leaky/models' --result_dir='stargan_new_6_leaky/result' --rafd_image_dir='stargan_new_6_leaky/data/train' --sample_dir='stargan_new_6_leaky/samples' --sample_label_dir='stargan_new_6_leaky/samples'
  ```

  

* 3 modes for this model

  1. Change face attribution(ex. Smile, sad, ..) for one person(cropped image) - person

     ```cmd
     python3 main.py --mode test  --test_iters 350000 --test_mode person
     ```

     

  2. Change face attribution for multi person(No need to crop) - group

     ```cmd
     python3 main.py --mode test  --test_iters 350000 --test_mode group
     ```

     

  3. Change face attribution for one person(No need to crop) - origin_person

  ```Cmd
  python3 main.py --mode test  --test_iters 350000 --test_mode origin_person
  ```

  Inferenced results are saved in args.result_dir -> user can set the directory


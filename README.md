# StarGAN-TA(StarGAN with Transfer Learning and changed Activation function)

* 3 modes for this model

  1. Change face attribution(ex. Smile, sad, ..) for one person(cropped image) - person
  2. Change face attribution for multi person(No need to crop) - group
  3. Change face attribution for one person(No need to crop) - origin_person

  ```Cmd
  python3 main.py --mode test  --test_iters 350000 --test_mode (person, group, origin_person)
  ```

  Inferenced results are saved in args.result_dir - user can set the directory


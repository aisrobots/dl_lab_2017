# dl_lab_2017
The exercises for the deep learning lab course we are teaching at the University of Freiburg.  The course page itself can be found here: http://ais.informatik.uni-freiburg.de/teaching/ws17/deep_learning_course/

## Running experiments
1. Connect to the main server
      ```Shell
      ssh username@login.informatik.uni-freiburg.de
      ```
2. Do not run anything on the main server. Connect to one of the pool 
  computer there instead. Preferably use tfpool25-46 or tfpool51-63 they provide the best GPUs.
      ```Shell
      ssh tfpoolXX
      ```
3. Replace XX with a 2 digit number of the computer you want to connect . 
Before running make sure no one else is using the selected computer. 
Also make sure your tensorflow program is running on the gpu and not the cpu.
      ```Shell
      who
      top
      nvidia-smi
      ```
4. Start a screen session
      ```Shell
      screen -S my_training
      python3
      import tensorflow
      ```
5. Detach from screen using: ctrl+a+d  
6. Login back into screen
      ```Shell
      screen -ls
      screen -r my_training
      ```
7. Write down on which computer you started your screen session.

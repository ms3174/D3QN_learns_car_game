# Using a Dueling Double Deep Q-Learning Approach with Prioritised Memory to Learn a 2D Racing Game

Created during one month project for CM50270 at the University of Bath in Spring 2022


## Compatible packages
- pygame (version 2.1.2)
- tensorflow2 (version 2.8)

## Guide

### Preparation

The pygame window is automatically sized to 1440x1440. If this is larger than the screen used:<br />
1.) Go to globals.py<br />
2.) Change screen_scaling to <.65<br />

Note: This might affect the behaviour of the car object and positioning of reward gates.

### Running main.py

Executing main.py in its original form will run the greedy policy of agent "32x16_double" (Eps. 679).<br />
Its trained weights are located in "/saved_models/selection/".<br />

Note: If globals.py does not define correct project path, change this manually to where main.py is located.

#### Running agent trainer

To start training of an agent (fixed at 32x16 neurons):<br />
1.) Choose the desired properties in main.py before "Window" class<br />
2.) Comment out "screen.test_run(dir_to_weights, text_to_display, epsilon_greedy_ON)" in "if __name__" loop<br />
3.) Uncomment "screen.train_run(num_training_eps)" in "if __name__" loop<br />

The agent trainer will train over the desired training period, and save trained weights every 50 episodes into "./saved_models/cache/" including the performance pickle file.

#### Running game with user input

To "play" the game by hand, uncomment "screen.user_baseline_run()" in "if __name__" loop

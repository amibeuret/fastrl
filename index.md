# FASTRL: a reinforcement learning exploration and benchmark

**FASTRL** is a a set of benchmark tasks used in surgical training which are adapted to the reinforcement learning setting for the purpose of training agents capable of providing assistance to the surgical trainees. The benchmark is provided with the purpose of exploring the domain of human-centric teaching agents within the learning-to-teach formalism.

**FASTRL** uses the FAST (Fundamentals of Arthroscopic Surgery Training) simulator provided by Virtamed AG 
to evaluate the RL pipeline. The simulation provides a number of educational navigation and manipulation tasks performed within a hollow dome structure in accordance with the Fundamentals of Arthroscopic Surgery Training (FAST) training problem developed by major American orthopaedic associations ABOS (American Board of Orthopaedic Surgery), AAOS (American Academy of Orthopaedic Surgeons) and AANA (Arthroscopy Association of North America).

# Benchmark Tasks

The benchmark contains three tasks: *ImageCentering* consisting of image centering and horizoning, *Periscoping* and *TraceLines* consisting of line tracing. These tasks consist of guiding the tip of the virtual endoscope to various locations marked by an avatar displaying visual cues. The goal is to orient the endoscope in such a way as to comply with the cues and center the image of the avatar in the field of view of the endoscope camera.

## *ImageCentering*
In *ImageCentering* the tip of the arthroscope (the Agent) must visualize the target avatar by centering the target in the field of view of the arthroscope. 

The video below demonstrates what a potential solution shoud look like. In this video, an agent is trained using a PPO policy with a carefully shaped reward function adapted to the task.

<iframe width="1113" height="631" src="https://www.youtube.com/embed/aBoQ87usZwQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## *TraceLines*
In *TraceLines*, the avatar will follow multiple splines, during which the tip of the arthroscope (the Agent) must follow the avatar. 

The video below demonstrates what a potential solution shoud look like. In this video, an agent is trained using a PPO policy with carefully shaped reward function adapted to the task.

<iframe width="1113" height="631" src="https://www.youtube.com/embed/SgtkEJP1-WY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



## *Periscoping*

In *Periscoping* the tip of the arthroscope (the Agent) must visualize the target avatar using angeld optics. 

The video below demonstrates what a potential solution shoud look like. In this video, an agent is trained using a PPO policy with carefully shaped reward function adapted to the task.

<iframe width="1113" height="629" src="https://www.youtube.com/embed/W1VmZ9HGg0A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


# Training an RL Agent Using ML-Agents

 The Unity Machine Learning Agents Toolkit ([ML-Agents](https://github.com/Unity-Technologies/ml-agents)) is an open source toolkit which provides PyTorch implementation of many known RL algorithms such as PPO, SAC and GAIL. 

If you are not familar with ML-Agents or not sure how to isntall it via pip, please visit their website.

The following command shows how to train a simple PPO model (described in the config file `configs/fast_ppo_config.yaml` provided in the repo) for the *Periscoping* environment (`--env pe/pe.x86_64`also provided in the repo)

```
mlagents-learn configs/fast_ppo_config.yaml --env builds/pe/pe.x86_64 --results-dir path/to/results --run-id my_experiment_name
```

To learn more about different provided environment please see [Environments](#environments)

# Training an RL Agent Using Stable Baselines3

[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) is another popular RL open source project providing a set of reliable implementations of reinforcement learning algorithms in PyTorch. 

Please check the Stable Baselines3 (SB3) website if you are not familiar with SB3 and for more information on how to setup SB3 via pip.

The following is an example of how to train a PPO agent (described in the config file) for the *ImageCentering* environment.
```
python sb3_main.py --ml_config_path configs/config_sb3.yaml --env builds/ic/ic.x86_64 --run_id my_experiment_name --results_dir results --task_name ImageCentering

```
## Training Configs
In the folder `configs` we provide template config files to train the agents. Depending on whether using ML-Agents or SB3 the config file must respect the required format.

# Environments

The environments are released as binaries, however we provide an API to interact with the environments of the dataset. A fixed set of parameters allow modification of key aspects of the environment. The parameters can be passed in a .yaml configuration file as well as through the command line. For the input to be interpreted correctly, especially the .yaml files, strict adherance to the layout and formatting of the input is necessary. 

Below is a list of all the binary files provided:

| Task              | MacOs   |  Linux | Windows |  Description |
|-------------------|---------|--------|---------|--------------|
| *ImageCentering*  | FAST_IC | ic     |   -     | Default binaries for *ImageCentering* task |
|  *Periscoping*    | FAST_PE | pe     |   -     | Default binaries for *Periscoping* task |
|  *TraceLines*     | FAST_TL | tl     |   -     | Default binaries for *TraceLines* task |


## Configurable parameters

The parameters that can be modified are categorzied in three groups: agent parameters, companion parameters, environment parameters. The agent parameters set values in either the agent class or the reward handler class. The companion parameters modify values the companion manager class and the environment parameters modify values the game manager class. 

For each task, the default parameters are provided in the `configs` folder.

### Agent parameters

The following table illustrates the configurable parameters pertaining to the agent. The only parameter that varies between different tasks is the list of reward weights due to the reward shaping being specific to each task.

| Parameter  | Type &ensp; | Description  |
|------------|----------|---------|
| name       | string   |shorthand name for the task currently being executed |
| [recordDemo](https://docs.unity3d.com/Packages/com.unity.ml-agents@1.0/api/Unity.MLAgents.Demonstrations.DemonstrationRecorder.html#Unity_MLAgents_Demonstrations_DemonstrationRecorder_Record) | bool     | whether a demonstration should be recorded in ML-Agents format|
| [maxStep](https://docs.unity3d.com/Packages/com.unity.ml-agents@1.0/api/Unity.MLAgents.Agent.html#Unity_MLAgents_Agent_MaxStep)    | int      | the maximum amount of steps the agent can take to complete a task before the episode is reset (from ML-Agents Agent class) |
| timeScale  | int      | Defines the multiplier for the deltatime in the simulation. If set to a higher value, time will pass faster in the simulation but the physics may perform unpredictably. Default 20 |
| rewardWeights | Dictionary<string, float> | the weights used to calculate the reward for a particular action; this list varies between tasks, see lists below |
|actionSpaceForce| bool | Wehther to use the default Force and Torque action space. (Must be true for most trianing scenarios) |
|newStateSpec| bool | Whether to use the new larger state space |
|stackSize| int | Stack size number to use in ML-Agents |
|goalConditioning| bool | Whether to include the target's positions in the state space|
|useCameraRotation| bool | Whether to rotate the camera |
|rotateTarget| bool | Whether to rotate the Target|

Reward weight lists for every task:
- Image Centering: 
    - progressBar
    - taskCompleted
    - leftDome
    - timeStepPenalty
    - maxVelocityPenalty
    - distReward
    - angleReward
    - visReward
- Periscoping:
    - progressBar
    - taskCompleted
    - leftDome
    - timeStepPenalty
    - maxVelocityPenalty
    - distReward
    - angleReward
    - raysReward
- TraceLines:
    - progressBar
    - reachedFocus
    - taskCompleted
    - leftDome
    - timeStepPenalty
    - maxVelocityPenalty
    - anglePenalty
    - distReward
    - angleReward
    - visReward

### Companion parameters

The following table illustrates the configurable parameters pertaining to the companion.

| Parameter  | Type &ensp;&ensp;    |Description  |
|------------|----------|--------|
| firstNPos  | int      | how many positions from the list of positions should be used in an episode, ie how many tasks per episode; should be in the range [1, positions.length] |
| positions  | List\<Vector3\> | the positions of the companion in the dome |
| randomPos  | bool     | whether to randomize the positions of the companion or not; if this is set, the positions list is disregarded |
| shufflePos | bool     | whether to shuffle the list of positions |
| addNoise   | bool     | whether to add noise to the positions |

To increase the number of positions in the positions list, it is not sufficient to increase the firstNPos parameter. Just increasing firstNPos will result in firstNPos being set (back) to positions.length, as firstNPos respects the invariant firstNPos <= positions.length. Instead, additional positions must be added manually with increasing indices (see section on configuration files).

The addNoise option will add random noise in the range [-0.3, 0.3] to all positions. Afterwards all the positions must be within the dome. If a position lies outside the dome, then the process is started over. The process can restart a maximum of 10 times. After these 10 tries, the fixed list of positions is taken (i.e. the positions with no noise are used).

The randomPos option will generate random positions respecting the following ranges for the coordinates:
- x: [-5.0,2.0]
- y: [4.0,8.0]
- z: [-1.5,1.0]

The generated positions will all be inside the dome, however it can happen that the arthroscope has difficulty reaching some of these postions.

### Environment parameters

The following table illustrates the configurable parameters pertaining to the environment.

| Parameter  | Type    |Description  |
|------------|---------|--------|
| seed       | int     | the random seed used to set UnityEngine.Random |
| model      | string  | the name of the trained model to run in inference mode |

The seed initializes [UnityEngine.Random](https://docs.unity3d.com/ScriptReference/Random.html) to allow non-deterministic behavior in the environment, such as generating random positions for the companion.

When running the environment in inference mode, the model (the 'brains' of the agent) can be changed. There is a default, built-in model that will run when no other model is specified. Otherwise, the given model will be run. The model must be stored in the '.onnx' format and must be placed in the Resources folder of the build. Just the name of the file (i.e. no extension) should be passed as an argument to load in the model.

## Configuration files

Input can be passed to the environment in a .yaml configuration file. There is a default config file that is always loaded into the environment to initialize all the parameters. This file should not be modified, as any discrepancies can cause the file to not be loaded, resulting in the environment crashing. To modify any of the values, pass them on through a custom configuration file. The custom config file will overwrite the default values. Here is the default configuration file for *ImageCentering* with the default values for all parameters:

    #tasks
    ImageCentering: 
      name: ic
      rewardWeights:
        progressBar: 0.2
        taskCompleted: 100.0
        leftDomeWeight: -1000.0
        timeStepPenalty: 0.0
        maxVelocityPenalty: 0.1
        distReward: 0.01
        angleReward: 0.001
        visReward: 0.01
      timeScale: 20
      recordDemo: false
      maxStep: 10000
      actionSpaceForce: true
      newStateSpec: false
      stackSize: 4
      goalConditioning: true
      useCameraRotation: true
      rotateTarget: false

    #companion
    firstNPos: 6
    positions:
      - #0:
        x: -2.0
        y: 4.5
        z: 0.91
      - #1:
        x: -5.0
        y: 4.5
        z: -1.09
      - #2:
        x: 0.5
        y: 8.0
        z: -0.09
      - #3:
        x: -2.0
        y: 4.0
        z: -0.09
      - #4:
        x: -5.0
        y: 6.0
        z: -1.09
      - #5:
        x: 1.0
        y: 7.5
        z: -0.09
    randomPos: false
    shufflePos: false
    addNoise: false
    noiseValue: 0.3
    randLoX: -3.0
    randHiX: 0.0
    randLoY: 2.0
    randHiY: 10.0
    randLoZ: -1.5
    randHiZ: 1.5
    sphereConstraint: false
    sphereCenter:
      x: 0.0
      y: 7.7
      z: 0.0
    sphereRadius: 2.0
    splines: #leave empty
    maxSplinePos: 0

    #other
    seed: 13 

The layout follows the standard layout for .yaml files.

### Custom configuration files

Custom configuration files can refer to any subset of parameters. However, it is very important that the layout adheres to the same layout in the default configuration file. The file should consist of three main blocks, delimited by the comments "#tasks", "#companion" and "#other". Any agent parameters should appear under the "#tasks" block, any companion parameters under the "#companion" block and environment parameters belong under the "#other" block. 

There is a global custom configuration file that is always loaded in, and during training this should be used to modify the default values. This is because ML-Agents does not allow command line input to the environment. Otherwise, when running the environment with a trained agent, both the given global config file and any .yaml config file passed via command line will be read in. The global config will overwrite the default values, and a file explicitly passed in the command line will overwrite the global config values. If the global custom config file should not be used or have any effect, leave it empty. 

All custom configuration files must be placed in the StreamingAssets folder, which is located in the *_Data folder of the build.

Here is an example how to format a custom configuration file: 


    #tasks
    ImageCentering: 
      rewardWeights:
        progressBar: 0.2
        taskCompleted: 100.0
        leftDomeWeight: -1000.0
      timeScale: 1
      recordDemo: false
    Periscoping: 
      timeScale: 1
      recordDemo: true
      maxStep: 10000

    #companion
    firstNPos: 6
    positions:
      - #1:
        x: -2.0
        y: 3.5
        z: 1
      - #6:
        x: 0.5
        y: 4.0
        z: 0.91
    addNoise: true

    #other
    seed: 14
    model: ImageCentering59


Note that indentation is very important and no additional comments should be left in the file other than the comments delimiting the blocks and the comments specifying the index number of a position.

For the positions list: any position in the list from the default config file can be changed by specifying the index in the comment after the dash (see above). Furthermore, additional positions can be added to the list by specifying the index to be greater than the indices of the other positions in the list. In the example above, the second position (with index 6) would be added to the end of the existing list. 

See below for examples on how to call the environment and pass a custom config file.

## Command line flags

Input can also be given via the command line, however only when running the environment with a trained model (i.e. not when training). Here are the flags we provide and how to pass arguments with them:

- tasks only flags:  
    &ensp; `--id-rewardWeights name=val,name=val,...`  
        &ensp;&ensp; where  
            &ensp;&ensp;&ensp; id is the shorthand name for the task
            &ensp;&ensp;&ensp; name refers to a specific weight  
            &ensp;&ensp;&ensp; val is the corresponding value  
    &ensp; `--id-fieldName val`  
        &ensp;&ensp; where  
            &ensp;&ensp;&ensp; id is the shorthand name for the task
            &ensp;&ensp;&ensp; fieldName refers to any agent parameter except rewardWeights  
            &ensp;&ensp;&ensp; val is the corresponding value  
- companion only flags:  
    &ensp; `--cm-positions num=x,y,z;num=x,y,z;...`  
        &ensp;&ensp; where  
            &ensp;&ensp;&ensp; num is an int denoting which position in the list  
            &ensp;&ensp;&ensp; x,y,z are floats, the coordinates of the position  
    &ensp; `--cm-fieldName val`  
        &ensp;&ensp; where  
            &ensp;&ensp;&ensp; fieldName refers to any companion paramter except positions  
            &ensp;&ensp;&ensp; val is the corresponding value  
- other flags:  
    &ensp; `--flagName val`  
        &ensp;&ensp; where  
            &ensp;&ensp;&ensp; flagName is name of the parameter to change  
            &ensp;&ensp;&ensp; val is the corresponding value  

